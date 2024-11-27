"""
Database tools for cataloguing trained models and notebook evaluations/figures.

Written with the help of Claude 3.5 Sonnet.
""" 

from collections.abc import Sequence
from datetime import datetime
import logging
from pathlib import Path
from typing import Literal, Optional, Dict, Any, TypeVar
import hashlib
import json
import uuid

from alembic.migration import MigrationContext
from alembic.operations import Operations
import jax.tree as jt
from jaxtyping import PyTree
import matplotlib.figure as mplf
import plotly
import plotly.graph_objects as go
from sqlalchemy import (
    Boolean,
    Column, 
    DateTime, 
    Integer, 
    Float, 
    ForeignKey,
    JSON, 
    String, 
    create_engine, 
    inspect,
    or_,
)
from sqlalchemy.ext.hybrid import hybrid_property
from sqlalchemy.orm import (
    DeclarativeBase, 
    Mapped, 
    Session, 
    mapped_column, 
    relationship, 
    sessionmaker,
)
from sqlalchemy.sql.type_api import TypeEngine

from feedbax import is_type, save, tree_zip
from feedbax._io import arrays_to_lists
from feedbax._tree import (
    everyf,
    is_not_type,
    make_named_dict_subclass,
    make_named_tuple_subclass,
)

from rnns_learn_robust_motor_policies import (
    DB_DIR, 
    FIGS_BASE_DIR, 
    MODELS_DIR, 
    QUARTO_OUT_DIR,
    REPLICATE_INFO_FILE_LABEL,
    TRAIN_HISTORY_FILE_LABEL, 
)


MODELS_TABLE_NAME = 'models'
EVALUATIONS_TABLE_NAME = 'evaluations'
FIGURES_TABLE_NAME = 'figures'


logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
# Prevent alembic from polluting the console with routine migration logs
logging.getLogger('alembic.runtime.migration').setLevel(logging.WARNING)


class Base(DeclarativeBase):
    type_annotation_map = {
        dict[str, Any]: JSON,
        dict[str, str]: JSON,
        Sequence[str]: JSON,
        Sequence[int]: JSON,
    }


BaseT = TypeVar("BaseT", bound=Base)


class ModelRecord(Base):
    __tablename__ = MODELS_TABLE_NAME
    
    id: Mapped[int] = mapped_column(primary_key=True)
    hash: Mapped[str] = mapped_column(String, unique=True, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    origin: Mapped[str]
    is_path_defunct: Mapped[bool] = mapped_column(default=False)
    postprocessed: Mapped[bool] = mapped_column(default=False)
    has_replicate_info: Mapped[bool]
    version_info: Mapped[Optional[dict[str, str]]]
    
    # Explicitly define some parameter columns to avoid typing issues, though our dynamic column 
    # migration would handle whatever parameters the user happens to pass, without this.
    disturbance_type: Mapped[str]
    where_train_strs: Mapped[Sequence[str]]
    n_replicates: Mapped[int]
    n_batches: Mapped[int]
    save_model_parameters: Mapped[Sequence[int]]
    
    @hybrid_property
    def path(self):
        return get_hash_path(MODELS_DIR, self.hash)

    @hybrid_property
    def replicate_info_path(self):
        if self.has_replicate_info:
            return get_hash_path(MODELS_DIR, self.hash, suffix=REPLICATE_INFO_FILE_LABEL)
        else: 
            return None
    
    @hybrid_property
    def train_history_path(self):
        return get_hash_path(MODELS_DIR, self.hash, suffix=TRAIN_HISTORY_FILE_LABEL)
    

MODEL_RECORD_BASE_ATTRS = [
    'id', 'hash', 'created_at', 'origin', 'is_path_defunct', 'has_replicate_info'
]
    
    
class EvaluationRecord(Base):
    """Represents a single evaluation."""
    __tablename__ = EVALUATIONS_TABLE_NAME

    model = relationship("ModelRecord")
    figures = relationship("FigureRecord", back_populates="evaluation")
    
    id: Mapped[int] = mapped_column(primary_key=True)
    hash: Mapped[str] = mapped_column(unique=True, nullable=False)
    created_at: Mapped[datetime] = mapped_column(default=datetime.utcnow)
    origin: Mapped[Optional[str]]
    model_hash: Mapped[Optional[str]] = mapped_column(ForeignKey(f'{MODELS_TABLE_NAME}.hash'))
    archived: Mapped[bool] = mapped_column(default=False)
    archived_at: Mapped[Optional[datetime]] = mapped_column(nullable=True)
    version_info: Mapped[Optional[dict[str, str]]]
    
    @hybrid_property
    def figure_dir(self):
        return FIGS_BASE_DIR / self.hash
    

class FigureRecord(Base):
    """Represents a figure generated during evaluation."""
    __tablename__ = FIGURES_TABLE_NAME

    evaluation = relationship("EvaluationRecord", back_populates="figures")
    model = relationship("ModelRecord")
    
    id: Mapped[int] = mapped_column(primary_key=True)
    hash: Mapped[str] = mapped_column(unique=True, nullable=False)
    created_at: Mapped[datetime] = mapped_column(default=datetime.utcnow)
    evaluation_hash: Mapped[str] = mapped_column(ForeignKey(f'{EVALUATIONS_TABLE_NAME}.hash'))
    identifier: Mapped[str]
    figure_type: Mapped[str]
    saved_formats: Mapped[Sequence[str]]
    archived: Mapped[bool] = mapped_column(default=False)
    archived_at: Mapped[Optional[datetime]] = mapped_column(nullable=True)
    
    # This is redundant because it can be inferred from `evaluation_hash`, but we include it for convenience
    model_hash: Mapped[str] = mapped_column(ForeignKey(f'{MODELS_TABLE_NAME}.hash'))
    
    # These are also redundant, and can be inferred from `model_hash` and `evaluation_hash`
    disturbance_type: Mapped[str] = mapped_column(nullable=True)


TABLE_NAME_TO_MODEL = {
    mapper.class_.__tablename__: mapper.class_
    for mapper in Base.registry.mappers
}


def get_sql_type(value) -> TypeEngine:
    if isinstance(value, bool):
        return Boolean()
    elif isinstance(value, int):
        return Integer()
    elif isinstance(value, float):
        return Float()
    elif isinstance(value, str):
        return String()
    else:
        return JSON()


def update_table_schema(engine, table_name: str, columns: Dict[str, Any]):
    """Dynamically add new columns using Alembic operations."""
    Base.metadata.create_all(engine)
    
    # Get existing columns
    inspector = inspect(engine)
    existing_columns = {col['name'] for col in inspector.get_columns(table_name)}
    
    # Create Alembic context
    context = MigrationContext.configure(engine.connect())
    op = Operations(context)
    model_class = TABLE_NAME_TO_MODEL[table_name]
    
    # Add only new columns using Alembic operations
    for key, value in columns.items():
        if key not in existing_columns:
            column = Column(key, get_sql_type(value), nullable=True)
            setattr(model_class, key, column)
            op.add_column(table_name, column)
            
    Base.metadata.clear()             # Clear SQLAlchemy's cached schema
    Base.metadata.create_all(engine)  # Recreate tables with new schema
            

def init_db(db_path: str = "sqlite:///models.db"):
    engine = create_engine(db_path)
    Base.metadata.create_all(engine)
    
    inspector = inspect(engine)
    
    for table_name in inspector.get_table_names():
        existing_columns = inspector.get_columns(table_name)
        
        model = TABLE_NAME_TO_MODEL[table_name]
        
        for col in existing_columns:
            try:
                getattr(model, col['name'])
            except AttributeError:
                setattr(
                    model, 
                    col['name'], 
                    Column(col['name'], col['type'], nullable=col['nullable']),
                ) 
    
    Base.metadata.clear()             # Clear SQLAlchemy's cached schema
    Base.metadata.create_all(engine)  # Recreate tables with new schema
    
    return sessionmaker(bind=engine)()


def get_db_session(key: str = "main"):
    """Create a database session for the project database with the given key."""
    return init_db(f"sqlite:///{DB_DIR}/{key}.db")
    


def hash_file(path: Path) -> str:
    """Generate MD5 hash of file."""
    md5 = hashlib.md5()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(4096), b''):
            md5.update(chunk)
    return md5.hexdigest()


def generate_temp_path(directory: Path, prefix: str = "temp_", suffix: str = ".eqx") -> Path:
    """Generate a temporary file path."""
    return directory / f"{prefix}{uuid.uuid4()}{suffix}"


def save_with_hash(tree: PyTree, directory: Path, hyperparameters: Optional[Dict] = None) -> tuple[str, Path]:
    """Save object to temporary file, compute hash, and move to final location."""
    temp_path = generate_temp_path(directory)
    save(temp_path, tree, hyperparameters=hyperparameters)
    
    file_hash = hash_file(temp_path)
    final_path = get_hash_path(directory, file_hash)
    temp_path.rename(final_path)
    
    return file_hash, final_path


def query_model_records(
    session: Session,
    filters: Optional[Dict[str, Any]] = None,
    match_all: bool = True,
    exclude_defunct: bool = True,
) -> list[ModelRecord]:
    """Query model records from database matching filter criteria."""
    if filters is None:
        filters = {}
        
    if exclude_defunct:
        check_model_files(session)
        filters['is_path_defunct'] = False
        
    return query_records(session, ModelRecord, filters, match_all)


def query_records(
    session: Session,
    record_type: str | type[BaseT],
    filters: Optional[Dict[str, Any]] = None,
    match_all: bool = True,
) -> list[BaseT]:
    """Query records from database matching filter criteria.
    
    Args:
        session: Database session
        record_type: SQLAlchemy model class or table name to query
        filters: Dictionary of {column: value} pairs to filter by
        match_all: If True, return only records matching all filters (AND).
                  If False, return records matching any filter (OR).
    """
    model_class: type[BaseT] = get_model_class(record_type)
    query = session.query(model_class)
    
    if filters:
        conditions = [
            getattr(model_class, key) == value 
            for key, value in filters.items()
        ]
        
        if match_all:
            for condition in conditions:
                query = query.filter(condition)
        else:
            query = query.filter(or_(*conditions))
            
    return query.all()


def get_record(
    session: Session,
    record_type: str | type[BaseT],
    **filters: Any,
) -> Optional[BaseT]:
    """Get single record matching all filters exactly.
    
    Args:
        session: Database session
        record_type: SQLAlchemy model class or table name to query
        **filters: Column=value pairs to filter by
    
    Raises:
        ValueError: If multiple matches found or unknown table name
    """
    model_class: type[BaseT] = get_model_class(record_type)
    matches = query_records(session, model_class, filters)
    if not matches:
        return None
    if len(matches) > 1:
        raise ValueError(f"Multiple {model_class.__name__}s found matching filters: {filters}")
    return matches[0]


def get_model_record(
    session: Session,
    exclude_defunct: bool = True,
    **filters: Any,
) -> Optional[ModelRecord]:
    """Get single model record matching all filters exactly.
    
    Args:
        session: Database session
        exclude_defunct: If True, only consider models with accessible files
        **filters: Column=value pairs to filter by
    
    Returns:
        Matching record, or None if not found
        
    Raises:
        ValueError: If multiple matches found
    """
    if exclude_defunct:
        check_model_files(session)
        filters['is_path_defunct'] = False
        
    return get_record(session, ModelRecord, **filters)


def get_model_class(record_type: str | type[BaseT]) -> type[BaseT]:
    """Convert table name to model class if needed."""
    if isinstance(record_type, str):
        if record_type not in TABLE_NAME_TO_MODEL:
            raise ValueError(f"Unknown table name: {record_type}")
        return TABLE_NAME_TO_MODEL[record_type]
    return record_type


def get_hash_path(directory: Path, hash_: str, suffix: str = '', ext: str = '.eqx') -> Path:
    components = [hash_, suffix]
    return (directory / "_".join(c for c in components if c)).with_suffix(ext)


def check_model_files(
    session: Session,
    clean_orphaned_files: Literal['no', 'delete', 'archive'] = 'no',
) -> None:
    """Check model files and update availability status."""
    logger.info("Checking availability of model files...")
    
    try:
        records = session.query(ModelRecord).all()
        known_hashes = {record.hash for record in records}
        
        for record in records:
            model_file_exists = get_hash_path(MODELS_DIR, record.hash).exists()
            replicate_info_file_exists = get_hash_path(
                MODELS_DIR, record.hash, suffix=REPLICATE_INFO_FILE_LABEL,
            ).exists()
            
            if record.is_path_defunct and model_file_exists:
                logger.info(f"File found for defunct model record {record.hash}; restored")
            elif not record.is_path_defunct and not model_file_exists:
                logger.info(f"File missing for model {record.hash}; marked as defunct")
            
            record.is_path_defunct = not model_file_exists
            record.has_replicate_info = replicate_info_file_exists
        
        if clean_orphaned_files != 'no':
            archive_dir = MODELS_DIR / "archive"
            for file_path in MODELS_DIR.glob("*.eqx"):
                file_hash = file_path.stem 
                if file_hash not in known_hashes: 
                    if clean_orphaned_files == 'delete':
                        logger.info(f"Deleting orphaned file: {file_path}")
                        file_path.unlink()
                    elif clean_orphaned_files == 'archive':
                        logger.info(f"Moving orphaned file to archive: {file_path}")
                        file_path.rename(archive_dir / file_path.name)
                        
        
        session.commit()
        logger.info("Finished checking model files")
        
    except Exception as e:
        session.rollback()
        logger.error(f"Error checking model files: {e}")
        raise e


def save_model_and_add_record(
    session: Session,
    origin: str,
    model: Any,
    model_hyperparameters: Dict[str, Any],
    other_hyperparameters: Dict[str, Any],
    train_history: Optional[Any] = None,
    train_history_hyperparameters: Optional[Dict[str, Any]] = None,
    replicate_info: Optional[Any] = None,
    replicate_info_hyperparameters: Optional[Dict[str, Any]] = None,
    version_info: Optional[Dict[str, str]] = None,
) -> ModelRecord:
    """Save model files with hash-based names and add database record."""
    (
        model_hyperparameters, 
        train_history_hyperparameters, 
        replicate_info_hyperparameters,
    ) = arrays_to_lists(
        (
            model_hyperparameters, 
            train_history_hyperparameters, 
            replicate_info_hyperparameters,
        )
    )
    
    hyperparameters = model_hyperparameters | other_hyperparameters
    
    # Save model and get hash-based filename
    model_hash, model_path = save_with_hash(model, MODELS_DIR, hyperparameters)
    
    # Save associated files if provided
    if train_history is not None:
        assert train_history_hyperparameters is not None, (
            "If saving training histories, must provide hyperparameters for deserialisation!"
        )
        save(
            get_hash_path(MODELS_DIR, model_hash, suffix=TRAIN_HISTORY_FILE_LABEL), 
            train_history,
            hyperparameters=train_history_hyperparameters,
        )
        
    if replicate_info is not None:
        assert replicate_info_hyperparameters is not None, (
            "If saving training histories, must provide hyperparameters for deserialisation!"
        )
        save(
            get_hash_path(MODELS_DIR, model_hash, suffix=REPLICATE_INFO_FILE_LABEL), 
            replicate_info,
            hyperparameters=replicate_info_hyperparameters,
        )
        
    update_table_schema(session.bind, MODELS_TABLE_NAME, hyperparameters)    
    
    # Create database record
    model_record = ModelRecord(
        hash=model_hash,
        origin=origin,
        is_path_defunct=False, 
        has_replicate_info=replicate_info is not None,
        version_info=version_info,
        **hyperparameters,
    )
    
    # Delete existing record with same hash, if it exists
    existing_record = get_record(session, ModelRecord, hash=model_hash)
    if existing_record is not None:
        session.delete(existing_record)
        session.commit()
        logger.debug(f"Replacing existing model record with hash {model_hash}")
    
    session.add(model_record)
    session.commit()
    return model_record


def generate_eval_hash(
    model_hash: Optional[str], 
    eval_params: Dict[str, Any],
    origin: Optional[str] = None,
) -> str:
    """Generate a hash for a notebook evaluation based on model hash and parameters.
    
    Args:
        model_hash: Hash of the model being evaluated. None for training notebooks.
        eval_params: Parameters used for evaluation
    """
    eval_str = "_".join([
        f"{model_hash or 'None'}",
        f"{origin or 'None'}",
        f"{json.dumps(eval_params, sort_keys=True)}",
    ])
    return hashlib.md5(eval_str.encode()).hexdigest()


def add_evaluation(
    session: Session,
    model_hash: Optional[str],  # Changed from ModelRecord to Optional[int]
    eval_parameters: Dict[str, Any],
    origin: Optional[str] = None,
    version_info: Optional[dict[str, str]] = None,
) -> EvaluationRecord:
    """Create new notebook evaluation record.
    
    Args:
        session: Database session
        model_id: ID of the model used (None for training notebooks)
        origin: ID of the notebook being evaluated
        eval_parameters: Parameters used for evaluation
    """
    eval_parameters = arrays_to_lists(eval_parameters)
    
    # Generate hash from model_id (if any) and parameters
    eval_hash = generate_eval_hash(
        model_hash=model_hash,
        eval_params=eval_parameters,
        origin=origin,
    )
    
    # Migrate the evaluations table so it has all the necessary columns
    update_table_schema(session.bind, EVALUATIONS_TABLE_NAME, eval_parameters)
    
    figure_dir = FIGS_BASE_DIR / eval_hash
    figure_dir.mkdir(exist_ok=True)
    
    quarto_output_dir = QUARTO_OUT_DIR / eval_hash
    quarto_output_dir.mkdir(exist_ok=True)
    
    # Delete existing record with same hash, if it exists
    existing_record = get_record(session, EvaluationRecord, hash=eval_hash)
    if existing_record is not None:
        existing_record.created_at = datetime.utcnow()
        logger.debug(f"Updating timestamp of existing evaluation record with hash {eval_hash}")
        eval_record = existing_record
    else:
        eval_record = EvaluationRecord(
            hash=eval_hash,
            model_hash=model_hash,  # Can be None
            origin=origin,
            version_info=version_info,
            **eval_parameters,
        )
        session.add(eval_record)

    session.commit()
    return eval_record


def generate_figure_hash(eval_hash: str, identifier: str, parameters: Dict[str, Any]) -> str:
    """Generate hash for a figure based on evaluation, identifier, and parameters."""
    figure_str = f"{eval_hash}_{identifier}_{json.dumps(parameters, sort_keys=True)}"
    return hashlib.md5(figure_str.encode()).hexdigest()


def savefig(
    fig, 
    label, 
    fig_dir: Path,
    image_formats: Sequence[str],
    transparent=True, 
    **kwargs,
):        
    path = str(fig_dir / f"{label}") + ".{ext}"
    
    if isinstance(fig, mplf.Figure): 
        for ext in image_formats:       
            fig.savefig(
                path.format(ext=ext),
                transparent=transparent, 
                **kwargs, 
            )
    
    elif isinstance(fig, go.Figure):
        # This will use the `orjson` package (faster) if installed
        fig.write_json(path.format(ext='json'), engine="auto")

        for ext in image_formats:
            fig.write_image(path.format(ext=ext), scale=2, **kwargs)
    

def add_evaluation_figure(
    session: Session,
    eval_record: EvaluationRecord,
    figure: go.Figure | mplf.Figure,
    identifier: str,
    save_formats: Optional[str | Sequence[str]] = "png",
    **parameters: Any,
) -> FigureRecord:
    """Save figure and create database record with dynamic parameters.
    
    Args:
        session: Database session
        eval_record: Evaluation record with which the figure is associated
        figure: Plotly or matplotlib figure to save
        identifier: Unique label for this type of figure
        save_formats: The image types to save. 
        **parameters: Additional parameters that distinguish the figure
    """
    parameters = arrays_to_lists(parameters)
    
    # Generate hash including parameters
    figure_hash = generate_figure_hash(eval_record.hash, identifier, parameters)
    
    if isinstance(save_formats, str):
        save_formats = [save_formats]
    elif isinstance(save_formats, Sequence):
        save_formats = list(save_formats)
    elif save_formats is None:
        save_formats = []
    
    # We automatically assume JSON is saved iff it's a plotly figure
    if 'json' in save_formats:
        save_formats.remove('json')
    
    # Maybe the user passed an extension with a leading dot
    save_formats = [format.strip('.') for format in save_formats]
        
    if isinstance(figure, mplf.Figure):
        figure_type = 'matplotlib'
    elif isinstance(figure, go.Figure):
        figure_type = 'plotly'
    
    # Save figure in subdirectory with same hash as evaluation
    eval_record.figure_dir.mkdir(exist_ok=True)
    
    savefig(figure, figure_hash, eval_record.figure_dir, save_formats)
    
    # Update schema with new parameters
    update_table_schema(session.bind, FIGURES_TABLE_NAME, parameters)
    
    figure_record = FigureRecord(
        hash=figure_hash,
        evaluation_hash=eval_record.hash,
        model_hash=eval_record.model_hash,
        identifier=identifier,
        figure_type=figure_type,
        saved_formats=save_formats,
        **parameters,
    )

    # Replace existing record if it exists
    existing_record = get_record(session, FigureRecord, hash=figure_hash)
    if existing_record is not None:
        session.delete(existing_record)
        session.commit()
        logger.debug(f"Replacing existing figure record with hash {figure_hash}")
    
    session.add(figure_record)
    session.commit()
    return figure_record


def use_record_params_where_none(parameters: dict[str, Any], record: Base) -> dict[str, Any]:
    """Helper to replace `None` values in `parameters` with matching values from `record`.
    
    Will raise an error if `parameters` contains any keys that are not columns in the type of `record`.
    """
    return {
        k: getattr(record, k) if v is None else v
        for k, v in parameters.items()
    }
    

def archive_orphaned_records(session: Session) -> None:
    """Mark records as archived if their model references no longer exist."""
    logger.info("Checking for orphaned records...")
    
    try:
        # Get all existing model hashes
        model_hashes = {r.hash for r in session.query(ModelRecord).all()}
        
        # Find and archive orphaned evaluation records
        orphaned_evals = (
            session.query(EvaluationRecord)
            .filter(
                EvaluationRecord.model_hash.isnot(None),  # Skip training evals
                ~EvaluationRecord.model_hash.in_(model_hashes),
                EvaluationRecord.archived == False
            )
            .all()
        )
        
        if orphaned_evals:
            now = datetime.utcnow()
            for record in orphaned_evals:
                logger.warning(
                    f"Archiving evaluation {record.hash} - referenced model "
                    f"{record.model_hash} no longer exists"
                )
                record.archived = True
                record.archived_at = now
                
                # Also archive associated figures
                for figure in record.figures:
                    if not figure.archived:
                        figure.archived = True
                        figure.archived_at = now
                        
            session.commit()
        
        # Find and archive orphaned figure records
        orphaned_figures = (
            session.query(FigureRecord)
            .filter(
                ~FigureRecord.model_hash.in_(model_hashes),
                FigureRecord.archived == False
            )
            .all()
        )
        
        if orphaned_figures:
            now = datetime.utcnow()
            for record in orphaned_figures:
                logger.warning(
                    f"Archiving figure {record.hash} - referenced model "
                    f"{record.model_hash} no longer exists"
                )
                record.archived = True
                record.archived_at = now
            session.commit()
            
        if not (orphaned_evals or orphaned_figures):
            logger.info("No orphaned records found")
            
    except Exception as e:
        session.rollback()
        logger.error(f"Error archiving orphaned records: {e}")
        raise


RecordDict = make_named_dict_subclass("RecordDict")
ColumnTuple = make_named_tuple_subclass("ColumnTuple")


def record_to_dict(record: Base) -> dict[str, Any]:
    """Converts an SQLAlchemy record to a dict."""
    return RecordDict({
        col.key: getattr(record, col.key) 
        for col in inspect(record).mapper.column_attrs
    })


def _value_if_unique(x: tuple):
    # Assume that all members of the tuple are the same type, since they come from the same column
    if isinstance(x[0], list):
        x = tuple(map(tuple, x))  # type: ignore
    if len(set(x)) == 1:
        return x[0]
    else:
        return x
    
    
# TODO: Use a DataFrame instead
def records_to_dict(records: list[Base], collapse_constant: bool = True) -> dict[str, Any]:
    """Zips multiple records into a single dict."""
    records_dict = tree_zip(
        *[record_to_dict(r) for r in records], 
        is_leaf=everyf(is_type(dict, list), is_not_type(RecordDict)),
        zip_cls=ColumnTuple,
    )
    if collapse_constant:
        records_dict = jt.map(
            lambda x: _value_if_unique(x),
            records_dict,
            is_leaf=is_type(ColumnTuple),
        )
    return records_dict
        

    
def retrieve_figures(
    session: Session,
    model_parameters: Optional[dict] = None,
    evaluation_parameters: Optional[dict] = None,
    exclude_archived: bool = True,
    **figure_parameters
) -> tuple[list[go.Figure], list[tuple[FigureRecord, EvaluationRecord, ModelRecord]]]:
    """Retrieve figures matching the given parameters.
    
    Parameters can contain tuples to match multiple values (OR condition).
    
    Args:
        session: Database session
        model_parameters: Parameters to match in models table
        evaluation_parameters: Parameters to match in evaluations table
        exclude_archived: If True, exclude archived figures
        **figure_parameters: Parameters to match in figures table
        
    Returns:
        Tuple of (list of plotly figures, list of record tuples)
        Each record tuple contains (figure_record, evaluation_record, model_record)
    """
    # Start with base query joining all three tables
    query = (
        session.query(FigureRecord, EvaluationRecord, ModelRecord)
        .join(EvaluationRecord, FigureRecord.evaluation_hash == EvaluationRecord.hash)
        .join(ModelRecord, FigureRecord.model_hash == ModelRecord.hash)
    )
    
    if exclude_archived:
        query = query.filter(FigureRecord.archived == False)

    def add_filters(query, model, parameters):
        for param, value in parameters.items(): 
            if isinstance(value, tuple):
                query = query.filter(getattr(model, param).in_(value))
            else:
                query = query.filter(getattr(model, param) == value)
        return query

    # Add figure parameter filters
    query = add_filters(query, FigureRecord, figure_parameters)
            
    # Add model parameter filters
    if model_parameters:
        query = add_filters(query, ModelRecord, model_parameters)
    
    # Add evaluation parameter filters
    if evaluation_parameters:
        query = add_filters(query, EvaluationRecord, evaluation_parameters)
    
    figures = []
    records = []
    for record_tuple in query.all():
        figure_record, _, _ = record_tuple
        json_path = FIGS_BASE_DIR / figure_record.evaluation_hash / f"{figure_record.hash}.json"
        if json_path.exists():
            try:
                figures.append(plotly.io.read_json(json_path))
                records.append(record_tuple)
            except Exception as e:
                logger.warning(f"Failed to load figure {figure_record.hash}: {e}")
                
    # TODO: Report # of figures, associated with # of evaluations
    
    return figures, records
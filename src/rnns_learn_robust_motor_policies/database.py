"""
Database tools for cataloguing trained models and notebook evaluations/figures.

Written with the help of Claude 3.5 Sonnet.
"""

from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any
import hashlib
import json

from sqlalchemy import create_engine, Column, Integer, Float, String, DateTime, JSON, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import Session, relationship, sessionmaker


Base = declarative_base()


class TrainedModel(Base):
    __tablename__ = 'trained_models'
    
    id = Column(Integer, primary_key=True)
    hash = Column(String, unique=True, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    notebook_id = Column(String, nullable=False)  # e.g. "1-1"
    
    # Core hyperparameters as columns for easy querying
    disturbance_type = Column(String)
    feedback_noise_std = Column(Float)
    motor_noise_std = Column(Float)
    feedback_delay_steps = Column(Integer)
    
    # All hyperparameters stored as JSON
    hyperparameters = Column(JSON)
    
    # File paths
    model_path = Column(String)
    train_history_path = Column(String)
    replicate_info_path = Column(String)


class NotebookEvaluation(Base):
    """Represents a single evaluation of a notebook."""
    __tablename__ = 'notebook_evaluations'
    
    id = Column(Integer, primary_key=True)
    hash = Column(String, unique=True, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Which notebook was evaluated
    eval_notebook_id = Column(String, nullable=False)  # e.g. "1-2a"
    
    # Reference to the model used
    model_id = Column(Integer, ForeignKey('trained_models.id'))
    model = relationship("TrainedModel")
    
    # Parameters used for evaluation (as JSON to handle different params per notebook)
    eval_parameters = Column(JSON)
    
    # Output directory containing rendered notebook
    output_dir = Column(String)
    
    # Relationship to figures
    figures = relationship("EvaluationFigure", back_populates="evaluation")


class EvaluationFigure(Base):
    """Represents a figure generated during notebook evaluation."""
    __tablename__ = 'evaluation_figures'
    
    id = Column(Integer, primary_key=True)
    hash = Column(String, unique=True, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Reference to the evaluation this figure belongs to
    evaluation_id = Column(Integer, ForeignKey('notebook_evaluations.id'))
    evaluation = relationship("NotebookEvaluation", back_populates="figures")
    
    # Figure metadata
    identifier = Column(String)  # e.g. "center_out_sets/all_evals_single_replicate"
    file_path = Column(String)


def init_db(db_path: str = "sqlite:///models.db"):
    engine = create_engine(db_path)
    Base.metadata.create_all(engine)
    return sessionmaker(bind=engine)()


def hash_file(path: Path) -> str:
    """Generate MD5 hash of file."""
    md5 = hashlib.md5()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(4096), b''):
            md5.update(chunk)
    return md5.hexdigest()


def generate_temp_path(directory: Path, prefix: str = "temp_", suffix: str = ".eqx") -> Path:
    """Generate a temporary file path."""
    import uuid
    return directory / f"{prefix}{uuid.uuid4()}{suffix}"


def save_with_hash(obj: Any, directory: Path, hyperparameters: Optional[Dict] = None) -> tuple[str, Path]:
    """Save object to temporary file, compute hash, and move to final location."""
    temp_path = generate_temp_path(directory)
    save(temp_path, obj, hyperparameters=hyperparameters)
    
    file_hash = hash_file(temp_path)
    final_path = directory / f"{file_hash}.eqx"
    temp_path.rename(final_path)
    
    return file_hash, final_path


def query_model_entries(
    session: Session,
    filters: Optional[Dict[str, Any]] = None,
    match_all: bool = True,
) -> list[TrainedModel]:
    """Query model entries from database matching filter criteria.
    
    Args:
        session: Database session
        filters: Dictionary of {column: value} pairs to filter by
        match_all: If True, return only entries matching all filters (AND).
                  If False, return entries matching any filter (OR).
    
    Returns:
        List of matching TrainedModel entries
    """
    if not filters:
        return session.query(TrainedModel).all()
        
    query = session.query(TrainedModel)
    
    conditions = [
        getattr(TrainedModel, key) == value 
        for key, value in filters.items()
    ]
    
    if match_all:
        for condition in conditions:
            query = query.filter(condition)
    else:
        from sqlalchemy import or_
        query = query.filter(or_(*conditions))
        
    return query.all()


def get_model_entry(
    session: Session,
    **filters: Dict[str, Any]
) -> Optional[TrainedModel]:
    """Get single model entry matching all filters exactly.
    Raises ValueError if multiple matches found."""
    matches = query_model_entries(session, filters)
    if not matches:
        return None
    if len(matches) > 1:
        raise ValueError(f"Multiple models found matching filters: {filters}")
    return matches[0]


def add_model_entry(
    session: Session,
    models_dir: Path,
    model: Any,
    hyperparameters: Dict[str, Any],
    train_history: Optional[Any] = None,
    replicate_info: Optional[Any] = None,
) -> TrainedModel:
    """Save model files with hash-based names and add database entry."""
    
    # Save model and get hash-based filename
    model_hash, model_path = save_with_hash(model, models_dir, hyperparameters)
    
    # Save associated files if provided
    train_history_path = None
    if train_history is not None:
        _, train_history_path = save_with_hash(
            train_history, 
            models_dir,
            hyperparameters,
        )
        
    replicate_info_path = None
    if replicate_info is not None:
        _, replicate_info_path = save_with_hash(
            replicate_info,
            models_dir,
            hyperparameters,
        )
    
    # Create database entry
    model_entry = TrainedModel(
        hash=model_hash,
        disturbance_type=hyperparameters['disturbance_type'],
        feedback_noise_std=hyperparameters['feedback_noise_std'],
        motor_noise_std=hyperparameters['motor_noise_std'],
        feedback_delay_steps=hyperparameters['feedback_delay_steps'],
        hyperparameters=hyperparameters,
        model_path=str(model_path),
        train_history_path=str(train_history_path) if train_history_path else None,
        replicate_info_path=str(replicate_info_path) if replicate_info_path else None,
    )
    
    session.add(model_entry)
    session.commit()
    return model_entry


def generate_eval_hash(model_id: Optional[int], eval_params: Dict[str, Any]) -> str:
    """Generate hash for notebook evaluation based on model ID and parameters."""
    eval_str = f"{model_id or 'None'}_{json.dumps(eval_params, sort_keys=True)}"
    return hashlib.md5(eval_str.encode()).hexdigest()


def add_notebook_evaluation(
    session: Session,
    model_id: Optional[int],  # Changed from TrainedModel to Optional[int]
    eval_notebook_id: str,
    eval_parameters: Dict[str, Any],
    output_base_dir: Path,
) -> NotebookEvaluation:
    """Create new notebook evaluation entry.
    
    Args:
        session: Database session
        model_id: ID of the model used (None for training notebooks)
        eval_notebook_id: ID of the notebook being evaluated
        eval_parameters: Parameters used for evaluation
        output_base_dir: Base directory for outputs
    """
    # Generate hash from model_id (if any) and parameters
    eval_hash = generate_eval_hash(
        model_id=model_id,
        eval_params=eval_parameters
    )
    
    output_dir = output_base_dir / eval_notebook_id / eval_hash
    
    eval_entry = NotebookEvaluation(
        hash=eval_hash,
        eval_notebook_id=eval_notebook_id,
        model_id=model_id,  # Can be None
        eval_parameters=eval_parameters,
        output_dir=str(output_dir),
    )
    
    session.add(eval_entry)
    session.commit()
    return eval_entry


def add_evaluation_figure(
    session: Session,
    evaluation: NotebookEvaluation,
    figure: Any,
    identifier: str,
) -> EvaluationFigure:
    """Save figure and create database entry."""
    # Save figure in subdirectory with same hash as evaluation
    figure_dir = Path(str(evaluation.output_dir)) / "figures"
    figure_dir.mkdir(exist_ok=True)
    
    # Generate unique filename for this figure
    figure_hash = hashlib.md5(f"{evaluation.hash}_{identifier}".encode()).hexdigest()
    figure_path = figure_dir / f"{figure_hash}.png"
    
    # Save figure
    figure.write_image(figure_path)  # For plotly
    # figure.savefig(figure_path)  # For matplotlib
    
    figure_entry = EvaluationFigure(
        hash=figure_hash,
        evaluation_id=evaluation.id,
        identifier=identifier,
        file_path=str(figure_path),
    )
    
    session.add(figure_entry)
    session.commit()
    return figure_entry
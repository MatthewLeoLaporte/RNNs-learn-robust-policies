
# 2025-03-03

- [x] Add batch script which runs a given script based on a given YAML file  🗎 Next > Archive > Notebooks as scripts > Batch scripts
	- For that, we need a common interface, i.e. we won’t need a different batch script for part 1 versus part 2.
- [x] Add YAML files indicating the spreads of models to train for part 1 and 2 🗎 Next > Archive > Notebooks as scripts > Batch scripts
- [x] Decide between argparse, and `main` functions that take a bunch of parameters (probably the latter.) 🗎 Next > Archive > Notebooks as scripts > Conversion
- [x] Update the main loop in `post_training` 🗎 Next > Archive > Model-per-file serialisation > What would be better? > Challenges
- [x] Records in the eval table correspond to runs of an analysis notebook. The analysis may be across several single models. Currently, the eval and figure records each have a single foreign reference to the hash of the training run (i.e. model record). Instead, we would need several such references. I don’t think these can be kept as multiple foreign SQL keys. ~~**Instead, we may need some kind of [linking table](https://stackoverflow.com/a/20572207).**~~ (I’m not doing this, for now.) A simpler solution is just store the model hashes in a list-of-string column, and forego the foreign key features, which are mostly a convenience at this point. 🗎 Next > Archive > Model-per-file serialisation > What would be better? > Challenges
- [x] `setup_task_model_pairs` should only take a single `disturbance_std`, and should not return a `TrainStdDict` 🗎 Next > Archive > Model-per-file serialisation > What would be better? > Challenges
- [x] ~~Update `query_and_load_models`~~: currently we pass the values of some model record fields, and check them one-to-one to find a match. Thus `disturbance_std_load` is a list of float, which is matched exactly with a list of float in the “models” table. **Instead, any sequences found in `query_and_load_models` should be interpreted as “map over the models table, loading multiple records, over this spread of values”.** For now, I think it is sufficient that we assume that if any lists are present, they are all the same length, and that we’re only loading a single spread (though multiple hyperparameter values may vary across individual models).  🗎 Next > Archive > Model-per-file serialisation > What would be better? > Challenges
	- **No, this shouldn’t be in `query_and_load_models`**. Instead, we should query and return a single model, and then map over this function in the notebook itself. Thus any logic about loading/handling multiple models for the purpose of an analysis, is done in the notebook.
- [x] Double check that `save_model_and_add_record` is detecting existing records – not just generating a new hash and save a new model even if all the hps match 🗎 Next > Debris
- [x] Make wrapper for `go.Figure` that adds a “copy as png” and “copy as svg” buttons; this will save a lot of time 🗎 Next > Efficiency
	- See https://plotly.com/python/figurewidget-app/
- [x] Maybe save a separate model table record in the database, for each *separate* disturbance std during training. This will: 🗎 Next > Efficiency
	- Prevent us from saving the same trained model multiple times because e.g. once we trained three runs `[0, 0.1, 1]` and another time two runs `[0, 0.1]`.
	- Force us to use the multi-loading logic in the analyses notebooks (i.e. load db entry with std 0, and db entry with std 0.1, and compare); note that this kind of logic will be necessary anyway for some of the supplementary analyses (e.g. noise comparisons).
	- Note that the foreign refs in the eval and fig tables would change into sets/lists, which is probably as it should be.
- [x] 2-1 🗎 Next > Technical > Convert notebooks for part 1
- [x] 1-3 🗎 Next > Technical > Convert notebooks for part 1
- [x] Make sure we save *all* analysis figures in `AbstractAnalysis.save_figs`; currently we’re only saving a subset for `Measures_CompareTrainStdAndContext`, it seems 🗎 Next > Technical
	- This was because of an incorrect change I had just made to `save_figs`, which is now reverted
- [x] **In `types`, make the mapping algorithmic between custom dict types and the column names they map to. Thus `PertVarDict` keys correspond to `pert_var` column values.** 🗎 Next > Technical
	- **Use `LDict`**
	- Similarly, we can use the same system to automatically determine the axis labels for `get_violins`

# 2025-03-06

- [x] ~~Construction of the analysis graph might be too complicated; is there a way to make analysis classes cache their results for a particular input, without causing problems with JAX?~~ 🗎 Next > Debris
- [x] ~~`seed`/base `key` column in each of the db tables~~ 🗎 Next > Technical
- [x] ~~**In `AbstractAnalysis.save_figs`, format the dump filename based on `path_params` and not just an integer counter**~~ 🗎 Next
	- Nope; instead I just modified

# 2025-03-10

- [x] ~~Are the different thin lines in the effector plots from `context_inputs` showing replicate? Why are they so parallel?~~ Yeah, I was using `Effector_ByReplicate` instead of `Effector_ByEval` 🗎 Next

# 2025-03-11

- [x] ~~Stop using `tmp` in figures dir~~ 🗎 Next > Debris
- [x] Pass legend values to plotting functions 🗎 Next
- [x] Control coloraxis from analysis subclasses 🗎 Next
- [x] Fix mean curve calculation in `trajectories_2d`: if there are multiple batch dimensions, it will average them all 🗎 Next
	- e.g. if the reach direction is not the color axis, we’ll end up averaging over reach directions and getting

# 2025-04-13

- [x] Syntax for constructing subplots?  🗎 Next > Convert notebooks > `AbstractAnalysis`
- [x] Aligned vars plotting function for `context_pert` which plots all four conditions (+/- plant pert, +/- context pert) on the same figure (one per training std) 🗎 Next > Convert notebooks > `AbstractAnalysis`
- [x] **Syntax for combining traces from single plots** 🗎 Next > Convert notebooks > `AbstractAnalysis`
	- This is also a common operation: we don’t just want to compare across a single axis/variable – we want to show multiple axes of variation on the same plot, as in a 2x2 condition
	- This is a bit trickier, since it may depend on the specific plotting function being used, and how we would like to visually differentiate the plots; however we could assume that the user knows (or provide them info about) the plotting function so that they can pass the relevant kwargs
	- The general idea is to run `make_figs` multiple times with slightly different kwargs, and then merge the traces from the resulting pytrees
	- e.g. `AlignedTrajectories(...).merge_figs_by("some_other_level)` 
- [x] **Syntax for pre-stacking certain PyTree levels** 🗎 Next > Convert notebooks > `AbstractAnalysis`
	- e.g. `AlignedTrajectories` should not do this specifically; instead it just takes `colorscale_axis` and we should assume that the right data is already stacked in that axis
	- this is a common enough operation that I don’t want `StackLevel` to be a usual analysis class, or something
	- instead, we should be able to do something like `AlignedTrajectories(...).after_stacking("some_level")`
- [x] Map operations, in addition to combine operations 🗎 Next > Convert notebooks > `AbstractAnalysis`
	- See cursor chat “Enhancing figure operations in AbstractAnalysis”

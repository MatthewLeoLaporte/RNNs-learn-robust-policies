.PHONY: nb2md


nb2md:
	@jupytext --to ../notebooks/markdown//md notebooks/*.ipynb
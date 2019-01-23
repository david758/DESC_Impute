
## Deep Embedding Single-cell Clustering  with Imputation

Relevant notebook is pbmc3k.ipynb. Need to run "Preprocessing" and then the code under "Plot the clusters with DESC".

All modifications in desc/models/*

The new train function can be run with new parameters:

impute = {True, False}: Whether to include pretrained AE in final model.
loss = {"mse", "nb", "poisson"}: Which loss function to use (from DCA).
is_stacked = False: Always set to False. I have not updated the SAE to use alternative losses yet.

The original DESC is: impute=Fasle, loss="mse".





# post_processing


# post_processing_pipeline
## Pipeline
![pipeline](/images/Experiment_pipeline.jpg)
### TODOs
- [ ] fairness metrics picking
- [ ] fn delta barplotting
- [ ] check differences on minimal asso rules/overlapping rules
- [ ] rule by rule "consistency check"
- [ ] why model performances metrics are better on detRank
- [ ] understanding of outlier results
- [ ] provare fit_predict con detRank
- [ ] fix `main.py`: run outside `src` folder
- [ ] add a delta plot for target_distribution


### Completed ✓
- [x] Delta improvement for metrics
- [x] Use csv dataset 
- [x] Split graphs of classification and fairness metrics
- [x] Associacion rules discovery at the beginning
- [x] Add confidence and support to the association rules tables 
- [x] Standardize fairness metrics, i.e.[-1,1]  
- [x] Visualization of how many male/female earn more and less 50K before/after pp 
- [x] Add >50K rules visualization
- [x] Add >50K rules visualization
- [x] Association rules extraction from test set
- [x] Update graphs accordingly
- [x] Table with support and confidence of association rules from test, test_pred, test_pred_pp
- [x] Try 60/40 on train/test in order to have >10k tuples on test set
- [x] [un]Privileged should be set after dataset analysis
- [x] look for more fairness metrics
- [x] update pptx
- [x] fix fairness metrics outliers and inconsistency
- [x] consistency delta: how many rows violete the association rules
- [x] accuracy measure: how many predictions are different from the real values
- [x] accuracy measure: [0:1]
- [x] add support and confidence before and afert the transf to the consistency table
- [x] add a quality delta
- [x] check for balanced sampling (plot distributions)
- [x] complete the pipeline for the other pp methods
- [x] DeterministicRanking accept only up to 5882 dataset rows (Det_rank.py to do)
- [x] add info about how metrics/values/perce
- [x] add a fairness delta
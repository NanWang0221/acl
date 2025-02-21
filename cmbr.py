import json
import os
from nimare.dataset import Dataset
from nimare.meta.cbmr import CBMREstimator, CBMRInference

factors_test = ['modality_re']

def remove_null_trial_number(data, factor):
    for study in list(data.values()):
        for contrast in list(study["contrasts"].values()):
            if contrast["labels"].get(factor) is None:
                del study["contrasts"][contrast]
        if not study["contrasts"]:
            del study

for f in factors_test:
    with open('cbmr_data1.json', 'r') as file:
        data = json.load(file)

    remove_null_trial_number(data, f)

    with open('cbmr_data2.json', 'w') as file:
        json.dump(data, file, indent=4)

    dset = Dataset('cbmr_data2.json', target="mni152_2mm")
    
    cbmr = CBMREstimator(
        moderators=[f],
        spline_spacing=100,
        model=models.PoissonEstimator,
        penalty=False,
        lr=1e-2,
        tol=1e-2,
        device="cpu"
    )
    results = cbmr.fit(dataset=dset)

    inference = CBMRInference(device="cuda")
    inference.fit(result=results)

    contrast_name = results.estimator.moderators
    t_con_moderators = inference.create_contrast(contrast_name, source="moderators")
    contrast_result = inference.transform(t_con_moderators=t_con_moderators)

    print(contrast_result.tables["moderators_regression_coef"])
    print(contrast_result.tables["p_" + f])

# SMPL Model Files

## License Notice

The SMPL body model is © 2015 Max-Planck-Gesellschaft and is licensed for
**non-commercial research, education, and artistic purposes only**.

By downloading and using these files you agree to the
[SMPL Model License](https://smpl.is.tue.mpg.de/modellicense).

**These model files are NOT included in this repository.**
Each user must individually register and download them from the official source.

## Download

1. Register at https://smpl.is.tue.mpg.de (free, academic license)
2. Download **SMPL for Python — version 1.1.0** (includes neutral model)
3. Unzip and copy the neutral model file to this directory:

```bash
unzip SMPL_python_v.1.1.0.zip
cp smpl/models/basicModel_neutral_lbs_10_207_0_v1.0.0.pkl demo/smpl/
```

## Expected files (gitignored)

```
demo/smpl/
  basicModel_neutral_lbs_10_207_0_v1.0.0.pkl   (~50 MB)
```

## Usage in JOSH pipeline

The SMPL model is loaded via the drag-drop UI at `/phase4.html` (Tab 1: SMPL Upload),
or programmatically via `SMPLLoaderUI` in `demo/josh/models/smpl-loader-ui.ts`.

The pipeline uses the **neutral** model (`basicModel_neutral_lbs_10_207_0_v1.0.0.pkl`)
with the first 10 shape PCA components (betas[0..9]).

Fields read from the .pkl:
- `v_template`    — [6890, 3] mean template vertices
- `f`             — [13776, 3] triangle faces
- `shapedirs`     — [6890, 3, 10] shape blend shapes
- `posedirs`      — [6890, 3, 207] pose blend shapes
- `J_regressor`   — [24, 6890] joint regressor
- `kintree_table` — [2, 24] kinematic tree
- `weights`       — [6890, 24] LBS skinning weights

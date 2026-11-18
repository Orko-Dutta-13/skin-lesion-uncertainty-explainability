# Reproduce the project. On Windows without make, run the commands in the README / docs instead (or use Docker).
SMOKE = data.master_csv=data/synthetic_processed/master.csv data.image_size=64 data.num_workers=0 model.pretrained=false train.epochs=2 train.batch_size=16 train.amp=false
RUNS  = runs/baseline runs/baseline_s1 runs/baseline_s2
SHIFT = runs/shift runs/shift_s1 runs/shift_s2

.PHONY: install test smoke smoke-uncertainty smoke-shift smoke-all data train eval-val eval-test gradcam \
        calibration mc selective split-ood train-shift eval-shift shift drift

install:
	pip install -r requirements.txt
test:
	pytest -q

# ---- fake data: the whole pipeline in about a minute on CPU, proves your environment works ----------------------
smoke:
	python -m reliable_derm.data.synthetic --out data/synthetic_raw
	python -m reliable_derm.data.prepare --raw-dir data/synthetic_raw --out-dir data/synthetic_processed --resize 64 --workers 2
	python -m reliable_derm.train --config configs/baseline.yaml --device cpu --override $(SMOKE) out_dir=runs/smoke
	python -m reliable_derm.evaluate --ckpt runs/smoke/best.pt --split val --device cpu
	python -m reliable_derm.evaluate --ckpt runs/smoke/best.pt --split test --device cpu
	python -m reliable_derm.explain.gradcam --ckpt runs/smoke/best.pt --split test --per-class 1 --device cpu
smoke-uncertainty: smoke
	python -m reliable_derm.calibration --run-dirs runs/smoke --out reports/smoke/calibration
	python -m reliable_derm.mc_dropout --ckpt runs/smoke/best.pt --splits val test --n-passes 3 --device cpu --num-workers 0
	python -m reliable_derm.selective --run-dirs runs/smoke --out reports/smoke/selective
smoke-shift: smoke
	python -m reliable_derm.train --config configs/shift.yaml --device cpu --override $(SMOKE) out_dir=runs/smoke_shift
	python -m reliable_derm.evaluate --ckpt runs/smoke_shift/best.pt --split val --split-col split_ood --device cpu
	python -m reliable_derm.evaluate --ckpt runs/smoke_shift/best.pt --split test --split-col split_ood --device cpu
	python -m reliable_derm.evaluate --ckpt runs/smoke_shift/best.pt --split shift_test --split-col split_ood --device cpu
	python -m reliable_derm.shift --run-dirs runs/smoke_shift --oracle-dirs runs/smoke --n-boot 50 --out reports/smoke/shift
	python -m reliable_derm.drift --ckpt runs/smoke_shift/best.pt --n 30 --out reports/smoke/drift --device cpu
smoke-all: smoke-uncertainty smoke-shift

# ---- real data (docs/DATA.md first) --------------------------------------------------------------------------
data:
	python -m reliable_derm.data.prepare --raw-dir data/raw --out-dir data/processed --resize 256
split-ood:        # adds the split_ood column to an existing master.csv (already included when you run `make data` now)
	python -m reliable_derm.data.add_ood_split
train:
	python -m reliable_derm.train --config configs/baseline.yaml
	python -m reliable_derm.train --config configs/baseline.yaml --override seed=1 out_dir=runs/baseline_s1
	python -m reliable_derm.train --config configs/baseline.yaml --override seed=2 out_dir=runs/baseline_s2
eval-val:
	for r in $(RUNS); do python -m reliable_derm.evaluate --ckpt $$r/best.pt --split val; done
eval-test:        # final numbers: run once
	for r in $(RUNS); do python -m reliable_derm.evaluate --ckpt $$r/best.pt --split test; done
gradcam:
	python -m reliable_derm.explain.gradcam --ckpt runs/baseline/best.pt --split test --mode correct --per-class 2
	python -m reliable_derm.explain.gradcam --ckpt runs/baseline/best.pt --split test --mode wrong --per-class 2
	python -m reliable_derm.explain.sanity --ckpt runs/baseline/best.pt --split test --per-class 2
	python -m reliable_derm.explain.shap_explain --ckpt runs/baseline/best.pt --split test --per-class 4
calibration:
	python -m reliable_derm.calibration
mc:
	for r in $(RUNS); do python -m reliable_derm.mc_dropout --ckpt $$r/best.pt --splits val test --n-passes 20; done
selective:
	python -m reliable_derm.selective
train-shift:
	python -m reliable_derm.train --config configs/shift.yaml
	python -m reliable_derm.train --config configs/shift.yaml --override seed=1 out_dir=runs/shift_s1
	python -m reliable_derm.train --config configs/shift.yaml --override seed=2 out_dir=runs/shift_s2
eval-shift:
	for r in $(SHIFT); do for s in val test shift_test; do \
	  python -m reliable_derm.evaluate --ckpt $$r/best.pt --split $$s --split-col split_ood; done; done
shift:
	python -m reliable_derm.shift
drift:
	python -m reliable_derm.drift --ckpt runs/shift/best.pt

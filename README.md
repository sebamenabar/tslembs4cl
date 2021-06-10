# Embeddings learned through traditional supervised learning can be used for continual learning

## Data

The preprocessed omniglot is available at [here](https://drive.google.com/file/d/19obzioEQ19jLp4Bm2oR3tv9YePcddiIf/view?usp=sharing) and pretrained models [here](https://drive.google.com/file/d/1brBalw4PsS14qNhUN2NUAZyq7svfhYou/view?usp=sharing).

## Examples of training commands

- Train TSL single-branch without augmentation

`python src/train_supervised_rebuilt.py --logdir results/supervised/oml --seed 0 --exp_name oml --sorted_test 0 --model oml --image_size 84 --train_split train+val --test_split test --test_zero_weight 1 --test_keep_bias 1 --normalize_input --num_epochs 600 --test_interval 200`

- Train TSL single-branch with augmentation

`python src/train_supervised_rebuilt.py --logdir results/supervised/oml --seed 0 --exp_name oml --sorted_test 0 --model oml --image_size 84 --train_split train+val --test_split test --test_zero_weight 1 --test_keep_bias 1 --normalize_input --num_epochs 600 --test_interval 200 --augment`

- Train TSL neuromodulated without augmentation

`python src/train_supervised_rebuilt.py --logdir results/supervised/anml --seed 0 --exp_name anml_aug --sorted_test 0 --model anml --image_size 28 --train_split train+val --test_split test --test_zero_weight 1 --test_keep_bias 1 --normalize_input --num_epochs 700 --schedule 600=0.1 --test_interval 100`

- Meta train single-branch without augmentation with zero init

`python src/train_mrcl_rebuilt.py --logdir results/meta/oml_aug --exp_name oml_zero_reset_b4_outer_500k  --seed 0 --num_steps 500000 --test_interval 250000 --model oml --image_size 84 --inner_lr 3e-2 --outer_lr 1e-4 --test_zero_weight 1 --test_keep_bias 0 --train_split train+val --test_split test --train_reset zero --normalize_input --reset_before_outer_loop 1 --batch_size 1 --train_method oml --lr_sweep 0.0005 0.00085 0.001`

- Meta train neurmodulated without augmentation with zero init

`python src/train_mrcl_rebuilt.py --logdir results/meta/anml/ --exp_name anml_zero_aug_all  --seed 0 --num_steps 500000 --test_interval 100000 --model anml --image_size 28 --inner_lr 1e-1 --outer_lr 1e-3 --test_zero_weight 1 --test_keep_bias 1 --train_split train+val --test_split test --train_reset zero --normalize_input --reset_before_outer_loop 1 --batch_size 1 --train_method anml --lr_sweep 0.0005 0.00085 0.001`

- Meta train single-branch augment both supports and queries with zero init

`python src/train_mrcl_rebuilt.py --logdir results/meta/oml_aug --exp_name oml_zero_reset_b4_outer_500k  --seed 0 --num_steps 500000 --test_interval 250000 --model oml --image_size 84 --inner_lr 3e-2 --outer_lr 1e-4 --test_zero_weight 1 --test_keep_bias 0 --train_split train+val --test_split test --train_reset zero --normalize_input --reset_before_outer_loop 1 --batch_size 1 --train_method oml --lr_sweep 0.0005 0.00085 0.001 --augment all`


## Examples of evaluation commands

- Evaluate TSL-trained without shuffling

`python src/evaluate_supervised.py --model_dir results/supervised/oml/oml_0/ --model_filename model.dill --sort 1 --shuffle 0 --zero_weight 1 --keep_bias 0`

- Evaluate Meta-trained with shuffling

- `python src/evaluate_meta.py  --model_dir results/meta/anml/oml_zero_reset_b4_outer_500k/ --model_filename model.dill --sort 0 --shuffle 1 --zero_weight 1 --keep_bias 0`



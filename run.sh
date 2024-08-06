python experiment_script.py --dataset ngsim --approach fed --strategy acc --last_interval 10
python experiment_script.py --dataset ngsim --approach fed --strategy sli --last_interval 10

python experiment_script.py --dataset ngsim --approach fed --strategy acc --last_interval 10 --proximal_term 1.0
python experiment_script.py --dataset ngsim --approach fed --strategy sli --last_interval 10 --proximal_term 1.0

python experiment_script.py --dataset ngsim --approach cen --strategy acc --last_interval 10
python experiment_script.py --dataset ngsim --approach cen --strategy sli --last_interval 10

python experiment_script.py --dataset rt --approach fed --strategy acc --last_interval 42
python experiment_script.py --dataset rt --approach fed --strategy sli --last_interval 42

python experiment_script.py --dataset rt --approach fed --strategy acc --last_interval 42 --proximal_term 1.0
python experiment_script.py --dataset rt --approach fed --strategy sli --last_interval 42 --proximal_term 1.0

python experiment_script.py --dataset rt --approach cen --strategy acc --last_interval 42
python experiment_script.py --dataset rt --approach cen --strategy sli --last_interval 42

python experiment_script.py --dataset sfc --approach fed --strategy acc --last_interval 42
python experiment_script.py --dataset sfc --approach fed --strategy sli --last_interval 42

python experiment_script.py --dataset sfc --approach fed --strategy acc --last_interval 42 --proximal_term 1.0
python experiment_script.py --dataset sfc --approach fed --strategy sli --last_interval 42 --proximal_term 1.0

python experiment_script.py --dataset sfc --approach cen --strategy acc --last_interval 42
python experiment_script.py --dataset sfc --approach cen --strategy sli --last_interval 42

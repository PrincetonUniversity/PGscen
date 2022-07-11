#!/bin/bash

#SBATCH --job-name=get_enrg-scores
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=16G
#SBATCH --time=200


# collect command line arguments
while getopts :s: var
do
	case "$var" in
	  s)  scen_dir=$OPTARG;;
	  [?])  echo "Usage: $0 " \
	      "[-s] output scenario directory" \
			exit 1;;
	esac
done

if [ ! -d "$scen_dir" ];
then
  echo "given scenario output directory does not exist!"
  exit 1
fi

# create output directory; load licensed software and conda environment
module purge
module load anaconda3/2021.11
conda activate pgscen


mkdir -p $scen_dir/scores/logs/
SCRIPT_PATH=$(dirname $(realpath "$0"))

for scen_file in $( find $scen_dir/scens_*.p.gz );
do
  day_str=${scen_file##*scens_}
  day_str=${day_str%.p.gz}

  sbatch --job-name=calc_enrg-scores --time=65 --mem-per-cpu=8G \
         --wrap=" python $SCRIPT_PATH/compute_energy_scores.py \
                            $scen_file /projects/PERFORM/Vatic_Grids/RTS-GMLC " \
         --output=$scen_dir/scores/logs/slurm_${day_str}.out \
         --error=$scen_dir/scores/logs/slurm_${day_str}.err
done

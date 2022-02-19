#!/bin/bash
#SBATCH --job-name=t7k_scens
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=4G
#SBATCH --time=100


# default command line argument values
opt_str=""
joint_opt=""

# collect command line arguments
while getopts :o:n:m:a:j var
do
	case "$var" in
	  o)  out_dir=$OPTARG;;
	  n)  scen_count=$OPTARG;;
	  m)  min_limit=$OPTARG;;
	  a)  opt_str=$OPTARG;;
	  j)  joint_opt="--joint";;
	  [?])  echo "Usage: $0 " \
	      "[-i] input directory" \
	      "[-o] output directory" \
	      "[-n] how many scenarios to generate" \
	      "[-m] maximum time to run in minutes" \
			exit 1;;
	esac
done

if [ ! -d "$out_dir" ];
then
  echo "given output directory does not exist, create it before running this pipeline!"
  exit 1
fi

module purge
module load anaconda3/2021.5
conda activate pgscen

run_times=()
for rand in $( shuf -i 0-362 -n 5 );
do
  use_date=$( date -d "2018-01-02 + $rand day" '+%Y-%m-%d' )

  start_time=$(date +%s)
  pgscen $use_date 1 -o $out_dir -n $scen_count $joint_opt -p -v
  end_time=$(date +%s)

  run_times+=($( echo "$end_time - $start_time" | bc ))
done

IFS=$'\n'
sort_times=$( echo "${run_times[*]}" | sort -n )
min_time=$( echo "$sort_times" | head -n1 )
max_time=$( echo "$sort_times" | tail -n1 )

day_time=$(( max_time + (max_time - min_time) ))
task_size=$( printf %.0f $( bc <<< "$min_limit * 60 / ($day_time * 1.17)" ))
ntasks=$(( 363 / task_size + 1 ))
task_days=$(( 363 / ntasks + 1 ))
use_time=$( printf %.0f $( bc <<< "$task_days * $day_time * 1.13 / 60" ))

day_jobs=()
fmt_str='+%Y-%m-%d'
echo "Submitting $ntasks scenario generation jobs..."

for i in $( seq 1 $ntasks );
do
  day_str=$( date -d "2018-01-02 + $(( (i - 1) * task_days )) day" $fmt_str )

  max_days=$(( ($( date +%s -d "2018-12-31" ) - $( date +%s -d "$day_str" )) / 86400 ))
  use_days=$(( task_days < max_days ? task_days : max_days ))

  day_jobs+=($( sbatch --job-name=t7k_scens --time=$use_time $opt_str --mem-per-cpu=4G \
                       --wrap=" pgscen $day_str $use_days \
                                       -o $out_dir -n $scen_count $joint_opt -p -v " \
                       --parsable \
                       --output=$out_dir/slurm_${day_str}.out \
                       --error=$out_dir/slurm_${day_str}.err ))
done

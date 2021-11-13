#!/bin/bash

# This script creates RTS scenarios for the entire year. It first creates
# scenarios for five random days and measures the runtime of each; these
# times are then used to estimate how many days can be simulated on a single
# compute node within the given time limit. According to this, the days of
# the year are partitioned into blocks, and each block is run in parallel
# on its own compute node.
#
# Arguments:
#   -i  The directory where the RTS-GMLC repository has been checked out.
#   -o  The directory where output files should be stored. This directory must
#       already exist; any existing scenario files within it will NOT be
#       overwritten.
#   -n  The number of scenarios to generate.
#   -m  The maximum runtime for each Slurm job spawned by this script, in
#       minutes. Use smaller maximum runtimes to generate scenarios faster
#       at the expense of having to use more cluster jobs. Maximum runtimes
#       of 100-200 are reasonable if there are a lot of idle nodes and you
#       want scenarios generated quickly, whereas runtimes of 500-800 are
#       more suitable for having this pipeline run overnight.
#
# Example usages:
#   sh create_scenarios.sh -i <data-dir>/RTS-GMLC -o <scratch-dir>/rts_scens \
#                          -n 1000 -m 150
#
#   sbatch --output=<scratch-dir>/slurm-logs/scen-pipeline.out \
#          --error=<scratch-dir>/slurm-logs/scen-pipeline.err \
#          repos/PGscen/pgscen/rts_gmlc/create_scenarios.sh \
#          -i <data-dir>/RTS-GMLC -o <scratch-dir>/rts-scens_4k -n 4000 -m 800

#SBATCH --job-name=create_rts-scens
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=4G
#SBATCH --time=100


module purge
module load anaconda3/2021.5
conda activate pgscen

# collect command line arguments
while getopts i:o:n:m: var
do
	case "$var" in
	  i)  in_dir=$OPTARG;;
	  o)  out_dir=$OPTARG;;
	  n)  scen_count=$OPTARG;;
	  m)  min_limit=$OPTARG;;
	  [?])  echo "Usage: $0 " \
	      "[-i] input directory" \
	      "[-o] output directory" \
	      "[-n] how many scenarios to generate" \
	      "[-m] maximum time to run in minutes" \
			exit 1;;
	esac
done


# run time trials using five randomly chosen days
run_times=()
for rand in $( shuf -i 0-363 -n 5 );
do
  use_date=$( date -d "2020-01-01 + $rand day" '+%Y-%m-%d' )

  start_time=$(date +%s)
  pgscen-rts $use_date 1 $in_dir -o $out_dir -n $scen_count -v
  end_time=$(date +%s)

  run_times+=($( echo "$end_time - $start_time" | bc ))
done

# calculate a conservative estimate of the worst-case runtime of a single day
# and use that to decide how many days we can run on a single node
IFS=$'\n'
sort_times=$( echo "${run_times[*]}" | sort -n )
min_time=$( echo "$sort_times" | head -n1 )
max_time=$( echo "$sort_times" | tail -n1 )

day_time=$(( max_time + (max_time - min_time) ))
task_size=$( printf %.0f $( bc <<< "$min_limit * 60 / ($day_time * 1.17)" ))
ntasks=$(( 364 / task_size + 1 ))
task_days=$(( 364 / ntasks + 1 ))
use_time=$( printf %.0f $( bc <<< "$task_days * $day_time * 1.13 / 60" ))

# break the year into evenly-sized chunks and generate scenarios for each
# chunk using its own Slurm job
day_jobs=()
fmt_str='+%Y-%m-%d'

for i in $( seq 1 $ntasks );
do
  day_str=$( date -d "2020-01-01 + $(( (i - 1) * task_days )) day" $fmt_str )

  day_jobs+=($( sbatch --job-name=rts-scens --time=$use_time --mem-per-cpu=4G \
                       --wrap=" pgscen-rts $day_str $task_days \
                                           $in_dir -o $out_dir -n $scen_count -v " \
                       --parsable \
                       --output=$out_dir/slurm_${day_str}.out \
                       --error=$out_dir/slurm_${day_str}.err ))
done


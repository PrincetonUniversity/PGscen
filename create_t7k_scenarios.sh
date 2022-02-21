#!/bin/bash

# This script creates scenarios for the Texas-7k power grid system for the
# entire year (2018) for which all input datasets are available. It first
# generates scenarios for five randomly-chosen days; using the runtimes
# measured for these days it estimates the number of days that can be run
# on a single compute node within the given time limit. Based on this the
# days of the year are partitioned into blocks, with each block running in
# parallel on its own compute node.

# Note that this script can be run directly on the command line (ideally using
# an interactive compute node instead of a head node) or submitted as a Slurm
# task in its own right — see the example usages listed below.
#
# Arguments:
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
#   -a  An optional string specifying additional options to be passed to the
#       Slurm scheduler.
#   -j  Generate load and solar scenarios together using a joint model instead
#       of the default behaviour in which they are modeled separately.
#
# Example usages:
#   sh create_scenarios.sh -o <scratch-dir>/t7k_scens -n 1000 -m 150
#   sh create_scenarios.sh -o <scratch-dir>/t7k_scens -n 500 -m 400 -a '--partition=orfeus'
#
#   sbatch --output=<scratch-dir>/slurm-logs/scen-pipeline.out \
#          --error=<scratch-dir>/slurm-logs/scen-pipeline.err \
#          repos/PGscen/create_t7k_scenarios.sh \
#             -o <scratch-dir>/t7k-scens_4k -n 4000 -m 800 -j

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
	      "[-o] output directory" \
	      "[-n] how many scenarios to generate" \
	      "[-m] maximum time to run pipeline in minutes" \
	      "[-a] additional Slurm scheduler options" \
	      "[-j] generate load and solar scenarios jointly?" \
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

# run time trials using five randomly chosen days
run_times=()
for rand in $( shuf -i 0-362 -n 5 );
do
  use_date=$( date -d "2018-01-02 + $rand day" '+%Y-%m-%d' )

  start_time=$(date +%s)
  pgscen $use_date 1 -o $out_dir -n $scen_count $joint_opt -p -v
  end_time=$(date +%s)

  run_times+=($( echo "$end_time - $start_time" | bc ))
done

# sort the time trial results and get the range of runtimes
IFS=$'\n'
sort_times=$( echo "${run_times[*]}" | sort -n )
min_time=$( echo "$sort_times" | head -n1 )
max_time=$( echo "$sort_times" | tail -n1 )

# calculate a conservative estimate of the worst-case runtime of a single day
# and use that to decide how many days we can run on a single node
day_time=$(( max_time + (max_time - min_time) ))
task_size=$( printf %.0f $( bc <<< "$min_limit * 60 / ($day_time * 1.17)" ))
ntasks=$(( 363 / task_size + 1 ))
task_days=$(( 363 / ntasks + 1 ))
use_time=$( printf %.0f $( bc <<< "$task_days * $day_time * 1.13 / 60" ))

# make sure we don't end up on the testing queue
if [ "$use_time" -le 61 ];
then
  use_time=62
fi

# break the year into evenly-sized chunks and generate scenarios for each
# chunk using its own Slurm job
day_jobs=()
fmt_str='+%Y-%m-%d'
echo "Submitting $ntasks scenario generation jobs..."

for i in $( seq 1 $ntasks );
do
  day_str=$( date -d "2018-01-02 + $(( (i - 1) * task_days )) day" $fmt_str )

  # make sure we don't try to generate scenarios for days past 2018-12-31
  max_days=$(( ($( date +%s -d "2018-12-31" ) - $( date +%s -d "$day_str" )) / 86400 ))
  use_days=$(( task_days < max_days ? task_days : max_days ))

  day_jobs+=($( sbatch --job-name=t7k_scens --time=$use_time $opt_str --mem-per-cpu=4G \
                       --wrap=" pgscen $day_str $use_days \
                                       -o $out_dir -n $scen_count $joint_opt \
                                       -p --skip-existing -v " \
                       --parsable \
                       --output=$out_dir/slurm_${day_str}.out \
                       --error=$out_dir/slurm_${day_str}.err ))
done

#! /bin/bash -e
in_dir=$1
out_dir=$2

function rename_pets ()
{
    #sed -e 's/^\([:alpha:]\+\)_.*\([0-9]\+\).*jpg$/\1_0000\2.jpg/' -
    #sed -e 's/^\(.+\)_.*[0-9].*jpg$/_0000.jpg/' -
    #sed "s/\(.*\)_ (\([0-9]\+\)).*.jpg/\1_0000\2.jpg/" -
    sed "s/\([a-zA-Z]\+\).\+(\([0-9]\+\))\.jpg/\1_0000\2.jpg/" -
    #sed -e 's/([:alpha:]*).jpg/_0000.jpg/' -
}
echo "In dir: $in_dir"
echo "Out dir: $out_dir"
mkdir -p $out_dir
for f in $in_dir/*.jpg
do 
    
    trimmed_f=$(echo $f | tr ' ' '_')
    #3echo "Trimmed $f to $trimmed_f"
    #echo "Basename: $(basename $trimmed_f)"
    #echo "f: $f"
    new_name="$(echo $(basename $trimmed_f) | rename_pets)"
    #echo $new_name
    echo "$f -> $out_dir/$new_name"
    cp "$f" "$out_dir/$new_name"
done
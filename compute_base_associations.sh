for ((i=0; i<50000; i=i+1000)); do
    end=$(($i + 999))
    python3 base_associations.py $i
done
for ((topIWK=1; topIWK<=10; topIWK=topIWK+1)); do
    printf "\nComputing test results for topIWK = $topIWK ...\n"
    python3 ./compute_image_links.py $topIWK
    python3 ./compute_partial_test_results.py 500 $topIWK
    printf "Successfully computed test results for topIWK = $topIWK.\n\n"
done
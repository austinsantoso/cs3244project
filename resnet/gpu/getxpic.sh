folderName="asl_alphabet_train"
# for each folder
for dir in "${folderName}/"*; do
    if [ -d "$dir" ]; then
        echo "${dir}"
        mkdir -p "copyOf/"${dir}
        for file in ${dir}"/"*0.jpg; do
            if [ -f "$file" ]; then
                # echo "Copying ./${file}"
                cp -r "$file" "copyOf/"${file}
            fi
        done
    fi
done
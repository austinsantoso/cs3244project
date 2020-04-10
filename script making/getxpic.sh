folderName="folder1"
# for each folder
for dir in "${folderName}/"*; do
    if [ -d "$dir" ]; then
        echo "${dir}"
        mkdir -p "copyOf/"${dir}
        for file in ${dir}"/"*0.txt; do
            if [ -f "$file" ]; then
                echo "Copying ./${file}"
                cp -r "$file" "copyOf/"${file}
            fi
        done
    fi
done
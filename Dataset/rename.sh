i=0
for file in image*; do
  mv "$file" "image$i.jpg"
  ((i++))
done


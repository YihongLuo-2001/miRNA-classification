for i in `ls ../../models_results/explanation/trees_dot/`
do
dot -Tsvg ../../models_results/explanation/trees_dot/$i -o ../../models_results/explanation/trees_svg/${i%.*}.svg
echo "$i -> ${i%.*}.svg"
done

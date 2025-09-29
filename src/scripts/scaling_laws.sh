#!/bin/bash

EPOCHS=(3 5 10 15 20)
NUM_LAYERS=(2 4 8)
HIDDEN_DIMS=(128 256 512)

OUTFILE="$1"

if [ -z "$OUTFILE" ]; then
    echo "Usage: $0 <outfile>"
    exit 1
fi

if [ ! -f "$OUTFILE" ]; then
    echo "Creating $OUTFILE..."
    touch $OUTFILE
fi

# Put header in OUTFILE
echo -e "MODEL SIZE (M)\tNUM LAYERS\tHIDDEN DIM\tEPOCHS\tTRAIN REGRESSION LOSS\tTRAIN CLASSIFICATION LOSS\n" >> $OUTFILE

for epochs in "${EPOCHS[@]}"; do
    for num_layer in "${NUM_LAYERS[@]}"; do
        for hidden_dim in "${HIDDEN_DIMS[@]}"; do
            echo "Parameters: epochs=$epochs, layers=$num_layer, hidden_dim=$hidden_dim"
            python src/trainer.py --results_file $OUTFILE --num_layers $num_layer --hidden_dim $hidden_dim --num_epochs $epochs
        done
    done
done
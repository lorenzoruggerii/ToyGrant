#!/bin/bash

EPOCHS=(10 15 20)
NUM_LAYERS=(4 8)
HIDDEN_DIMS=(128 256 512 1024)

LOSSDIR="$1"
PLOTDIR="$2"


if [ -z "$LOSSDIR" ]; then
    echo "Usage: $0 <lossdir> <plotdir>"
    exit 1
fi

if [ -z "$PLOTDIR" ]; then
    echo "Usage: $0 <lossdir> <plotdir>"
    exit 1
fi

# if [ ! -f "$OUTFILE" ]; then
#     echo "Creating $OUTFILE..."
#     touch $OUTFILE
# fi

# Put header in OUTFILE
# echo -e "MODEL SIZE (M)\tNUM LAYERS\tHIDDEN DIM\tEPOCHS\tTRAIN REGRESSION LOSS\tTRAIN CLASSIFICATION LOSS\n" >> $OUTFILE

for epochs in "${EPOCHS[@]}"; do
    for num_layer in "${NUM_LAYERS[@]}"; do
        for hidden_dim in "${HIDDEN_DIMS[@]}"; do
            echo "Parameters: epochs=$epochs, layers=$num_layer, hidden_dim=$hidden_dim"
            python src/trainer.py --loss_path ${LOSSDIR}/${epochs}E_${num_layer}L_${hidden_dim}H.tsv --num_layers $num_layer --hidden_dim $hidden_dim --num_epochs $epochs --use_pos_embs --lr 0.001 --bs 32 --plot_path ${PLOTDIR}/${epochs}E_${num_layer}L_${hidden_dim}H.png
        done
    done
done
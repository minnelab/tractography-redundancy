#!/bin/bash
# Run TractSeg bundle segmentation + endings segmentation for all subjects
# in Data/Diffusion/

DIFFUSION_DIR="/home/maia-user/tracto_work/Data/Diffusion"

for SUBJ in $(ls "$DIFFUSION_DIR"); do
    SUBJ_DIR="$DIFFUSION_DIR/$SUBJ/T1w"
    INPUT="$SUBJ_DIR/Diffusion/data.nii.gz"
    OUTPUT="$SUBJ_DIR/tractseg_output"
    BVALS="$SUBJ_DIR/Diffusion/bvals"
    BVECS="$SUBJ_DIR/Diffusion/bvecs"

    if [ -d "$OUTPUT/endings_segmentations" ]; then
        echo "=== $SUBJ: already complete, skipping ==="
        continue
    fi

    echo "=== $SUBJ: Running TractSeg bundle segmentation ==="
    TractSeg -i "$INPUT" \
        --bvals "$BVALS" \
        --bvecs "$BVECS" \
        --raw_diffusion_input \
        -o "$OUTPUT"

    if [ $? -ne 0 ]; then
        echo "=== $SUBJ: Bundle segmentation FAILED ==="
        continue
    fi

    echo "=== $SUBJ: Running TractSeg endings segmentation ==="
    TractSeg -i "$OUTPUT/peaks.nii.gz" \
        -o "$OUTPUT" \
        --output_type endings_segmentation

    if [ $? -ne 0 ]; then
        echo "=== $SUBJ: Endings segmentation FAILED ==="
        continue
    fi

    echo "=== $SUBJ: COMPLETE ==="
done

echo "All subjects processed."

mkdir -p frames

#for vid in ./videos/*.mp4; do
#    name=$(basename "$vid" .mp4)
#    mkdir -p frames/v_$name
#    ffmpeg -i "$vid" -q:v 2 frames/v_$name/img_%05d.jpg
#done

#mkdir -p frames_out

for vid in ./videos/*.mp4; do
    name=$(basename "$vid" .mp4)
    mkdir -p frames/v_$name
    ffmpeg -i "$vid" -vf "scale=340:256" -q:v 2 frames/v_$name/img_%05d.jpg
done

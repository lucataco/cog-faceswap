# faceswap Cog model

This is semi-custom implementation of faceswap as a Cog model, based on the original code of [roop](https://github.com/s0md3v/roop). [Cog packages machine learning models as standard containers.](https://github.com/replicate/cog)

First, download the pre-trained weights:

    cog build -t faceswap

Then, you can run predictions:

    cog predict -i target_image=@tony.jpg -i swap_image=@elon.jpg

## Example Output

Example output for sample inputs:

![alt text](output.jpg)

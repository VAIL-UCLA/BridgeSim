from PIL import Image
import glob

x = 533
y = 300

fp_in = '/home/abhijit/Work/BridgeSim/waymo_rendered_output/camera/raster_f0/*.jpg'
fp_out = '/home/abhijit/Work/BridgeSim/waymo_rendered_output/camera/raster_f0/video.gif'
q = 10 # Quality
img, *imgs = [Image.open(f).resize((x,y),Image.Resampling.LANCZOS) for f in sorted(glob.glob(fp_in))] 
img.save(fp=fp_out, format='GIF', append_images=imgs,quality=q, 
         save_all=True, duration=15, loop=0, optimize=True)
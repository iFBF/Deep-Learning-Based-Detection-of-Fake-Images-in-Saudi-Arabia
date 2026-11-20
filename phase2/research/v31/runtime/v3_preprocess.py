"""Portable deterministic preprocessing matching the reviewed author transform."""
from io import BytesIO
from PIL import Image,ImageOps
import hashlib
import numpy as np


def decode(data):
    if not data or len(data)>10*1024*1024:raise ValueError('Use an image file of at most 10 MiB.')
    with Image.open(BytesIO(data)) as raw:
        if raw.format not in {'JPEG','PNG','WEBP'}:raise ValueError('Supported formats: JPEG, PNG, WEBP.')
        if getattr(raw,'is_animated',False) or getattr(raw,'n_frames',1)>1:raise ValueError('Use a single still image.')
        if raw.width*raw.height>12_000_000 or min(raw.size)<16:raise ValueError('Input dimensions exceed the original service contract.')
        image=ImageOps.exif_transpose(raw).convert('RGB');image.load()
        return image,{'width':image.width,'height':image.height,'decoded_format':raw.format,
                      'sha256':hashlib.sha256(data).hexdigest(),'exif_orientation_applied':True}


def preprocess(image):
    image=ImageOps.exif_transpose(image).convert('RGB');w,h=image.size
    width,height=(440,int(440*h/w)) if w<=h else (int(440*w/h),440)
    image=image.resize((width,height),resample=Image.Resampling.BILINEAR)
    left=int(round((width-384)/2.));top=int(round((height-384)/2.))
    image=image.crop((left,top,left+384,top+384))
    x=np.asarray(image,dtype=np.float32).transpose(2,0,1)/np.float32(255.)
    x=(x-np.asarray([.485,.456,.406],dtype=np.float32)[:,None,None])/np.asarray([.229,.224,.225],dtype=np.float32)[:,None,None]
    return np.ascontiguousarray(x[None])

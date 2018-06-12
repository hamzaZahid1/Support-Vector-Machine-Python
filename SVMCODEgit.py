
import rasterio
from rasterio.mask import mask
import geopandas as gpd

from sklearn.model_selection import train_test_split
from sklearn.svm import SVC 
import gdal

import numpy.ma as ma
from sklearn.svm import LinearSVC
import numpy as np

shapefile = gpd.read_file("sample1path")
# extract the geometry in GeoJSON format
geoms = shapefile.geometry.values # list of shapely geometries
geometry = geoms[0] # shapely geometry
# transform to GeJSON format
from shapely.geometry import mapping
geoms = [mapping(geoms[0])]
# extract the raster values values within the polygon 
with rasterio.open("tif-file-path") as src:
     out_image1, out_transform = mask(src, geoms, crop=True)
     




shapefile = gpd.read_file("sample2path")
# extract the geometry in GeoJSON format
geoms = shapefile.geometry.values # list of shapely geometries
geometry = geoms[0] # shapely geometry
# transform to GeJSON format
from shapely.geometry import mapping
geoms = [mapping(geoms[0])]
# extract the raster values values within the polygon 
with rasterio.open("tif-file-path") as src:
     out_image3, out_transform = mask(src, geoms, crop=True)
     


nonwaterarray= np.transpose((ma.resize(out_image3, (4, out_image3.shape[1]*out_image3.shape[2])))) #converted3darrayto2darray
waterarray= np.transpose((ma.resize(out_image1, (4, out_image1.shape[1]*out_image1.shape[2]))))

nonwaterarray=nonwaterarray[~(nonwaterarray==0).all(1)]
waterarray=waterarray[~(waterarray==0).all(1)]


xTrain=np.concatenate((nonwaterarray,waterarray)).astype('f')

yTrain=np.concatenate((0*np.ones(nonwaterarray.shape[0]),np.ones(waterarray.shape[0])))


src_ds=gdal.Open("tif-file-path") 
print ("Size of X Pixel: {0}".format(src_ds.RasterXSize))
print ("Size of Y Pixel: {0}".format(src_ds.RasterYSize))
rb1 = np.array(src_ds.GetRasterBand(1).ReadAsArray())
rb2 = np.array(src_ds.GetRasterBand(2).ReadAsArray())
rb3 = np.array(src_ds.GetRasterBand(3).ReadAsArray())
rb4 = np.array(src_ds.GetRasterBand(4).ReadAsArray())
[cols, rows] = rb1.shape
rb1=np.ravel(rb1)
rb2=np.ravel(rb2)
rb3=np.ravel(rb3)
rb4=np.ravel(rb4)

xPredict=np.hstack((rb1[:,None],rb2[:,None],rb3[:,None],rb4[:,None])).astype('f')


#clf = SVC(C=100.0, cache_size=8000,kernel='poly')
clf = LinearSVC(C=1)
clf.fit(xTrain, yTrain)
#numS=clf.support_vectors_
outy = clf.predict(xPredict)

outyy = outy.reshape(cols,rows)
outyy = outyy.astype('uint16')
#io.imshow(palette[supervised])



driver = gdal.GetDriverByName('GTiff')
out_ds = driver.Create('outputtif',rows, cols, 1, gdal.GDT_UInt16)
out_ds.SetGeoTransform(src_ds.GetGeoTransform())
out_ds.SetProjection(src_ds.GetProjection())
out_ds.GetRasterBand(1).WriteArray(outyy)
out_ds.FlushCache() ##saves to disk!!
out_ds = None


from matplotlib import pyplot as plt
plt.imshow(outyy, interpolation='nearest')
plt.show()
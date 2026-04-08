import ee
import pandas as pd
import os

try:
    # Initialize ee without arguments, assuming it's authenticated
    ee.Initialize()
except Exception:
    pass

def query_gee_ndvi(lat, lon, start_date, end_date):
    """
    Query Sentinel-2 Surface Reflectance imagery for NDVI and return mean score
    """
    try:
        # Re-ensure initialization if needed, though typically handled at module load
        try:
            ee.Initialize()
        except Exception:
            pass
        
        point = ee.Geometry.Point([lon, lat])
        collection = (ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
                      .filterBounds(point)
                      .filterDate(start_date, end_date)
                      .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 20)))
        
        count = collection.size().getInfo()
        if count == 0:
            raise Exception("No image found for the given criteria.")
            
        def calculate_image_ndvi(image):
            # NDVI = (NIR - Red) / (NIR + Red) where NIR is B8 and Red is B4 for Sentinel-2
            ndvi = image.normalizedDifference(['B8', 'B4']).rename('NDVI')
            return image.addBands(ndvi)
        
        ndvi_collection = collection.map(calculate_image_ndvi)
        mean_ndvi_img = ndvi_collection.select('NDVI').mean()
        
        mean_dict = mean_ndvi_img.reduceRegion(
            reducer=ee.Reducer.mean(),
            geometry=point,
            scale=10,
            maxPixels=1e9
        ).getInfo()
        
        if mean_dict and 'NDVI' in mean_dict and mean_dict['NDVI'] is not None:
            return float(mean_dict['NDVI'])
        raise Exception("NDVI calculation returned null")
    except Exception as e:
        print(f"GEE Fetch Error: {e}")
        return None

def fallback_ndvi_from_csv(lat, lon):
    try:
        # Simple fallback reading the 'dataset.csv' if available 
        # that already contains mock data
        csv_path = os.path.join(os.path.dirname(__file__), 'dataset.csv')
        df = pd.read_csv(csv_path)
        if 'NDVI' in df.columns:
             # we could match by nearest lat/lon, but for simplicity returning the mean or the median
             return float(df['NDVI'].mean())
    except Exception as e:
        print(f"CSV Fallback failed: {e}")
    # Hard fallback
    return 0.55

def get_live_ndvi(lat, lon, start_date, end_date):
    ndvi_val = query_gee_ndvi(lat, lon, start_date, end_date)
    if ndvi_val is not None:
        return {"ndvi": ndvi_val, "source": "Google Earth Engine"}
    
    fallback_val = fallback_ndvi_from_csv(lat, lon)
    return {"ndvi": fallback_val, "source": "dataset.csv (fallback)"}

def calculate_ndvi(red, nir):
    """
    Calculate Normalized Difference Vegetation Index (NDVI).
    NDVI = (NIR - Red) / (NIR + Red)
    """
    try:
        red = float(red)
        nir = float(nir)
    except (ValueError, TypeError):
        return 0.0

    if (nir + red) == 0:
        return 0.0
    
    ndvi = (nir - red) / (nir + red)
    return ndvi

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from osgeo import gdal, ogr, osr
import os
import glob

# =========================
# TARGET GRID REQUIREMENTS
# =========================
TARGET_RES = 10            # meters / pixel (usato per definire l'estensione del quadrato)
TARGET_SIZE = 10980        # pixels (width = height)
TARGET_EXTENT = TARGET_RES * TARGET_SIZE   # 109800 meters
HALF_EXTENT = TARGET_EXTENT / 2            # 54900 meters

FORCED_NODATA = 0          # <-- come richiesto


def utm_epsg_from_lonlat(lon: float, lat: float) -> int:
    """EPSG UTM WGS84: 326xx (Nord) / 327xx (Sud)."""
    zone = int((lon + 180) / 6) + 1
    return (32600 + zone) if lat >= 0 else (32700 + zone)


def read_union_geometry_and_srs(shp_path: str):
    ds = ogr.Open(shp_path)
    if ds is None:
        raise RuntimeError(f"Impossibile aprire shapefile: {shp_path}")

    layer = ds.GetLayer(0)
    srs = layer.GetSpatialRef()
    if srs is None:
        raise ValueError("CRS dello shapefile non definito (manca .prj?)")

    union_geom = None
    for feat in layer:
        geom = feat.GetGeometryRef()
        if geom is None:
            continue
        geom = geom.Clone()
        union_geom = geom if union_geom is None else union_geom.Union(geom)

    ds = None

    if union_geom is None or union_geom.IsEmpty():
        raise ValueError("Geometria shapefile vuota")

    return union_geom, srs


def build_target_grid_from_shp(shp_path: str):
    """
    Costruisce la griglia target (CRS metrico):
      - se shp è in gradi: sceglie UTM zona corretta e trasforma la geometria per ottenere il centro in metri
      - se shp è già proiettato: usa quel CRS
    Output:
      dst_wkt, bounds[xmin,ymin,xmax,ymax]
    """
    geom, shp_srs = read_union_geometry_and_srs(shp_path)

    if shp_srs.IsGeographic():
        c = geom.Centroid()
        lon, lat = c.GetX(), c.GetY()
        epsg = utm_epsg_from_lonlat(lon, lat)

        dst_srs = osr.SpatialReference()
        dst_srs.ImportFromEPSG(epsg)

        print(f"[INFO] Shapefile in gradi -> target UTM EPSG:{epsg}")

        ct = osr.CoordinateTransformation(shp_srs, dst_srs)
        geom2 = geom.Clone()
        geom2.Transform(ct)
        cc = geom2.Centroid()
        cx, cy = cc.GetX(), cc.GetY()
    else:
        dst_srs = shp_srs.Clone()
        cc = geom.Centroid()
        cx, cy = cc.GetX(), cc.GetY()

    xmin = cx - HALF_EXTENT
    xmax = cx + HALF_EXTENT
    ymin = cy - HALF_EXTENT
    ymax = cy + HALF_EXTENT

    bounds = [xmin, ymin, xmax, ymax]
    return dst_srs.ExportToWkt(), bounds


def describe_raster(ds: gdal.Dataset, label: str):
    proj = ds.GetProjection()
    gt = ds.GetGeoTransform(can_return_null=True)
    print(f"[DBG] {label}: size={ds.RasterXSize}x{ds.RasterYSize} bands={ds.RasterCount}")
    print(f"[DBG] {label}: projection={'OK' if proj else 'MISSING'}")
    print(f"[DBG] {label}: geotransform={'OK' if gt else 'MISSING'}")
    if proj:
        print(f"[DBG] {label}: proj head={proj.splitlines()[0] if proj.splitlines() else proj[:80]}")
    if gt:
        print(f"[DBG] {label}: GT={gt}")


def align_raster_to_shp(
    input_raster: str,
    shp_path: str,
    output_raster: str,
    assume_src_epsg_if_missing: int = 4326,
    assert_size: bool = True
):
    """
    Warpa input_raster su griglia definita dallo shp:
      - target CRS: UTM (se shp era in gradi) oppure CRS shp
      - bounds quadrati fissi (109800m)
      - dimensioni FORZATE: 10980 x 10980  (evita 10981x10981)
      - nodata FORZATO: 0
    """
    dst_wkt, bounds = build_target_grid_from_shp(shp_path)

    src_ds = gdal.Open(input_raster, gdal.GA_ReadOnly)
    if src_ds is None:
        raise RuntimeError(f"Impossibile aprire raster: {input_raster}")

    describe_raster(src_ds, "SRC")

    # Se manca CRS nel raster, assumo EPSG:4326 (gradi)
    src_proj = src_ds.GetProjection()
    if not src_proj:
        src_srs = osr.SpatialReference()
        src_srs.ImportFromEPSG(assume_src_epsg_if_missing)
        src_wkt = src_srs.ExportToWkt()
        print(f"[WARN] Raster senza CRS -> assumo EPSG:{assume_src_epsg_if_missing}")
    else:
        src_wkt = None

    os.makedirs(os.path.dirname(output_raster), exist_ok=True)

    gdal.UseExceptions()

    # NB: niente targetAlignedPixels + niente xRes/yRes
    #     per evitare l'effetto "10981x10981"
    warp_opts = gdal.WarpOptions(
        format="GTiff",
        srcSRS=src_wkt,
        dstSRS=dst_wkt,
        outputBounds=bounds,            # [xmin, ymin, xmax, ymax]
        width=TARGET_SIZE,              # 10980
        height=TARGET_SIZE,             # 10980
        resampleAlg="near",
        srcNodata=FORCED_NODATA,
        dstNodata=FORCED_NODATA,
        multithread=True,
        warpOptions=["NUM_THREADS=ALL_CPUS"],
        creationOptions=["COMPRESS=LZW", "TILED=YES", "BIGTIFF=IF_SAFER"]
    )

    out_ds = gdal.Warp(output_raster, src_ds, options=warp_opts)

    src_ds = None
    if out_ds is None:
        msg = gdal.GetLastErrorMsg()
        raise RuntimeError(f"gdal.Warp fallito. Ultimo errore GDAL: {msg}")

    out_w, out_h = out_ds.RasterXSize, out_ds.RasterYSize
    print(f"[DBG] OUT size={out_w}x{out_h}")

    out_ds = None

    if assert_size and (out_w != TARGET_SIZE or out_h != TARGET_SIZE):
        raise RuntimeError(f"Output non ha dimensioni attese {TARGET_SIZE}x{TARGET_SIZE} ma {out_w}x{out_h}")


if __name__ == "__main__":

    raster_dir = "/home/silvia/Downloads/tiles"
    shapefile = "/home/silvia/Downloads/shp/22MCT/22MCT.shp"
    out_dir = "/home/silvia/Downloads/tiles_utm"

    os.makedirs(out_dir, exist_ok=True)

    rasters = sorted(glob.glob(os.path.join(raster_dir, "*.tif")) +
                    glob.glob(os.path.join(raster_dir, "*.tiff")))

    for in_ras in rasters:
        fname = os.path.basename(in_ras)
        out_ras = os.path.join(out_dir, fname)

        print(f"\n[DO] {fname}")
        try:
            align_raster_to_shp(
                input_raster=in_ras,
                shp_path=shapefile,
                output_raster=out_ras,
                assume_src_epsg_if_missing=4326,
                assert_size=True
            )
            print(f"[OK] -> {out_ras}")
        except Exception as e:
            print(f"[ERR] {fname}: {e}")

    print("\nRaster alignment complete.")

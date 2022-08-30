psql -U desi -h decatdb.lbl.gov desidb -c "\copy (select DISTINCT s.RA, s.DEC, s.TARGETID, fh.id, fh.tileid, fh.petal, fh.night, fhz.targetid, fhz.z, fhz.zerr, fhz.zwarn, fhz.spectype, fhz.subtype, fhz.deltachi2, fhz.cumultile_id, fhf.targetid, fhf.target_ra, fhf.target_dec, fhf.obsconditions, fhf.release, fhf.brickid, fhf.brick_objid, fhf.fiberflux_ivar_g, fhf.fiberflux_ivar_r, fhf.fiberflux_ivar_z, fhf.morphtype, fhf.flux_g, fhf.flux_r, fhf.flux_z, fhf.flux_ivar_g, fhf.flux_ivar_r, fhf.flux_ivar_z, fhf.ebv, fhf.flux_w1, fhf.flux_w2, fhf.flux_ivar_w1, fhf.flux_ivar_w2, fhf.fiberflux_g, fhf.fiberflux_r, fhf.fiberflux_z, fhf.fibertotflux_g, fhf.fibertotflux_r, fhf.fibertotflux_z, fhf.sersic, fhf.coadd_numexp, fhf.coadd_exptime, fhf.coadd_numnight, fhf.coadd_numtile, fhf.cumultile_id, pv.objid, pv.brickid, pv.brickname, pv.ra, pv.dec
From static.secondary as s
JOIN daily.tiles_redshifts as fhz ON s.TARGETID = fhz.targetid
JOIN daily.tiles_fibermap as fhf ON s.TARGETID = fhf.targetid 
JOIN static.pv as pv ON fhf.brickid = pv.brickid and fhf.brickname = pv.brickname and fhf.brick_objid = pv.objid
JOIN daily.cumulative_tiles as fh ON fh.id = fhf.cumultile_id
where fhz.SPECTYPE LIKE 'GAL%'
AND s.program LIKE 'PV%'
AND fhz.ZWARN = 0
AND pv.pvtype LIKE 'FP%'
)
TO 'daily_pv_targets_FP_per_tiles.csv' CSV HEADER;"
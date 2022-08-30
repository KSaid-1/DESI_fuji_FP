.open /global/cfs/cdirs/desi/science/td/db/desi.db
.headers on
.mode csv
.output pvfp.csv
SELECT DISTINCT s.PROGRAM, s.RA, s.DEC, s.TARGETID, zb.TILE, zb.YYYYMMDD, zb.PETAL, zb.PRODUCTION, zb.Z, zb.ZERR, zb.GROUPING, zb.SPECTYPE, rp.PRODUCTION, rp.Z, rp.ZERR, rp.SPECTYPE, rp.DELTACHI2, rp.ZWARN
FROM secondary as s
	JOIN zbest_denali as zb ON s.TARGETID = zb.TARGETID
	JOIN redshifts_prod as rp ON s.TARGETID = rp.TARGETID
WHERE  
	zb.SPECTYPE LIKE 'GAL%'
	AND s.PROGRAM = 'PV_DARK_HIGH'
	AND rp.COADD = 'cumulative'
        AND zb.GROUPING = 'cumulative'	
	AND zb.ZWARN = 0

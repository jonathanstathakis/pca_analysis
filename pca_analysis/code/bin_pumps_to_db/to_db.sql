    CREATE SCHEMA IF NOT EXISTS bin_pump;
    
    -- create the tables

    CREATE TABLE IF NOT EXISTS
        bin_pump.solvcomps (
        -- unique number per block corresponding to the tbl_num in bin_pump.id
        tbl_num INTEGER,
        idx INTEGER,
        channel VARCHAR,
        ch1_solv VARCHAR,
        name_1 VARCHAR,
        ch2_solv VARCHAR,
        name_2 VARCHAR,
        selected VARCHAR,
        used VARCHAR,
        percent FLOAT,
        );

    CREATE TABLE IF NOT EXISTS
        bin_pump.timetables (
        tbl_num INTEGER,
        idx INTEGER,
        time FLOAT,
        a FLOAT,
        b FLOAT,
        flow FLOAT,
        pressure FLOAT
    );

    -- create the primary key table to check against
    -- See: <https://stackoverflow.com/questions/72883083/create-an-auto-incrementing-primary-key-in-duckdb>
    CREATE SEQUENCE IF NOT EXISTS
        bin_pump_tbl_nums
    START 1;

    -- create the primary key table
    CREATE TABLE IF NOT EXISTS
        bin_pump.id (
            id VARCHAR PRIMARY KEY,
            tbl_num INTEGER UNIQUE DEFAULT NEXTVAL('bin_pump_tbl_nums')
            );

    COMMENT ON TABLE
        bin_pump.id
    IS
        'A primary key table for the solvcomps and timetable tables as they are stored in blocks. Use tbl_num to join the blocks to the ids for samplewise extraction';
    
    COMMENT ON COLUMN
        bin_pump.id.id
    IS
        'sample primary key. Use in association with tbl_num to get a samples solvcomp or timetable from those tables';

    COMMENT ON COLUMN
        bin_pump.id.tbl_num
    IS
        'the numerical order of the samples added to the tables in this schema (bin_pump). Used to extract a samples table from solvcomps or timetables through joining. i.e. join the sample to the id in this table, use this tbl_num to join to the data tables.';

    COMMENT ON TABLE
        bin_pump.timetables
    IS
        'The sample solvent timetables - information about the elution, whether it was gradient, etc. In wide format with samples stacked vertically, use tbl_num and id in bin_pump.id to extract samplewise.';

    COMMENT ON TABLE
        bin_pump.solvcomps
    IS
        'Information about a samples solvent composition. In wide format with samples stacked vertically, use tbl_num and id in bin_pump.id to extract samplewise.';
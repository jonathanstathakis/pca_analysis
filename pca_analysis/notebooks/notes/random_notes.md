# Random Notes

## DUCKDB SUMMARIZE APPROX IS APPROX

cdt: 2024-09-05T13:32:21

duckdb SUMMARIZE has an 'approx_unique' column which is exactly that, inaccurate. Don't rely on it. I did once, and wasted half an hour investigating the discrepency.
-- DATA CLEANING...

select * 
from world_layoff.layoffs_data;

select distinct company, count(*) over() as num_of_rows
from world_layoff.layoffs_data;

DROP TABLE IF EXISTS world_layoff.layoffs;

CREATE TABLE world_layoff.layoffs
LIKE world_layoff.layoffs_data;

SELECT *
FROM world_layoff.layoffs;

INSERT 
INTO world_layoff.layoffs 
select * 
from world_layoff.layoffs_data;

SELECT *
FROM world_layoff.layoffs;

SELECT 
	 DISTINCT company,
    (SELECT COUNT(company) FROM world_layoff.layoffs) AS total_rows,
    (SELECT COUNT(DISTINCT company) FROM world_layoff.layoffs) AS distinct_companies
FROM world_layoff.layoffs
ORDER BY 1;

UPDATE world_layoff.layoffs
SET company = TRIM(company),
	Location_HQ = TRIM(Location_HQ),
    Industry = TRIM(Industry),
    Country = TRIM(Country),
    Stage = TRIM(Stage),
    Date_Added = str_to_date(Date_Added, '%Y-%m-%d %H:%i:%s');
    
SELECT Date_Added, YEAR(Date_Added)
FROM world_layoff.layoffs;

SELECT *
FROM world_layoff.layoffs
WHERE Company LIKE '%Open';

ALTER TABLE world_layoff.layoffs
DROP COLUMN Date,
DROP COLUMN Source,
DROP COLUMN Percentage,
DROP COLUMN List_of_Employees_Laid_Off;

WITH CTE AS(
SELECT *,
		ROW_NUMBER() OVER (PARTITION BY Company, Location_HQ, Industry, Laid_Off_Count, YEAR(Date_Added), MONTH(Date_Added), DAY(Date_Added), Stage, Country) AS RN,
		ROW_NUMBER() OVER (ORDER BY Company, Location_HQ, Industry, Laid_Off_Count, YEAR(Date_Added), MONTH(Date_Added), DAY(Date_Added), Stage, Country) AS row_num
FROM world_layoff.layoffs)
SELECT * 
FROM CTE
WHERE RN>1;

SELECT * 
FROM world_layoff.layoffs
WHERE Company LIKE 'Zymergen';

DROP TABLE IF EXISTS world_layoff.DELETE_CTE;
CREATE TEMPORARY TABLE world_layoff.DELETE_CTE AS
	SELECT *,
		ROW_NUMBER() OVER (PARTITION BY Company, Location_HQ, Industry, Laid_Off_Count, YEAR(Date_Added), MONTH(Date_Added), DAY(Date_Added), Stage, Country) AS RN
			FROM world_layoff.layoffs;
    
SELECT * 
FROM world_layoff.DELETE_CTE
WHERE RN >1
ORDER BY 9 DESC;

ALTER TABLE world_layoff.layoffs
	ADD COLUMN RN INT;
    


DELETE
FROM world_layoff.layoffs
WHERE (Company, Location_HQ, Industry, Laid_Off_Count, Funds_Raised, Stage,Date_Added, Country) IN (
    SELECT Company, Location_HQ, Industry, Laid_Off_Count, Funds_Raised, Stage,Date_Added, Country
    FROM world_layoff.DELETE_CTE
		WHERE RN > 1
);

SELECT * 
FROM world_layoff.layoffs
WHERE Company LIKE 'Zymergen';

WITH CTE AS(
	SELECT *,
		ROW_NUMBER() OVER (PARTITION BY Company, Location_HQ, Industry, Laid_Off_Count, YEAR(Date_Added), MONTH(Date_Added), DAY(Date_Added), Stage, Country) AS RN
			FROM world_layoff.layoffs)
SELECT * 
FROM CTE
WHERE RN>1;

SELECT *
FROM world_layoff.layoffs
WHERE Company = 'Alerzo';

SELECT * 
FROM world_layoff.layoffs
WHERE Laid_Off_Count=''
ORDER BY 1;

DELETE 
	FROM world_layoff.layoffs
    WHERE Laid_Off_Count='';
    
SELECT Company, Laid_Off_Count
FROM world_layoff.layoffs
ORDER BY 1;

SELECT * 
FROM world_layoff.layoffs;

ALTER TABLE world_layoff.layoffs
	DROP COLUMN Funds_Raised;
    
ALTER TABLE world_layoff.layoffs
MODIFY COLUMN Laid_Off_Count INT,
MODIFY COLUMN Date_Added DATE;

SELECT * 
FROM world_layoff.layoffs;


WITH CTE AS(
SELECT *, 
	ROW_NUMBER() OVER (PARTITION BY Company, Location_HQ, Industry, Stage, YEAR(Date_Added), MONTH(Date_Added), Country) AS RN
FROM world_layoff.layoffs)

SELECT * 
FROM CTE 
WHERE RN>1
;

SELECT *
FROM world_layoff.layoffs
WHERE Company = 'Amazon'
ORDER BY 6;

SELECT * 
FROM world_layoff.layoffs
ORDER BY 1;



-- DATA ANALYSING...
SELECT Country, SUM(Laid_Off_Count)
FROM world_layoff.layoffs
GROUP BY Country
ORDER BY 2 DESC;


SELECT Country, YEAR(Date_Added) AS 'YEAR', SUM(Laid_Off_Count) AS 'LAID OFF COUNT'
FROM world_layoff.layoffs
GROUP BY Country, YEAR(Date_Added)
ORDER BY 3 DESC;

SELECT Country, SUM(Laid_Off_Count) AS 'LAID OFF COUNT OF 2024'
FROM world_layoff.layoffs
WHERE YEAR(Date_Added) = 2024
GROUP BY Country
ORDER BY 2 DESC;

SELECT Country, SUM(Laid_Off_Count) AS 'LAID OFF COUNT OF 2023'
FROM world_layoff.layoffs
WHERE YEAR(Date_Added) = 2023
GROUP BY Country
ORDER BY 2 DESC;

SELECT Country, MONTH(Date_Added) AS 'MONTH', SUM(Laid_Off_Count) AS 'LAID OFF COUNT OF 2023'
FROM world_layoff.layoffs
WHERE YEAR(Date_Added) = 2023 
GROUP BY Country, MONTH(Date_Added)
ORDER BY 2 ASC, 3 DESC;

SELECT 
	Country, 
	YEAR(Date_Added), 
	MONTH(Date_Added) AS 'MONTH', 
	SUM(Laid_Off_Count) AS 'LAID OFF COUNT OF 2023'
FROM world_layoff.layoffs
	GROUP BY Country, YEAR(Date_Added), MONTH(Date_Added)
	ORDER BY 2, 3, 4 DESC;
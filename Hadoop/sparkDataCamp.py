#CHAPTER 1 SPARK BASICS
----------------------------------------------
from pyspark.sql import SparkSession

# Create a SparkSession 
my_spark = SparkSession.builder.getOrCreate()

# Print my_spark
print(my_spark)

----------------------------------------
#see tables in the data inside spark cluster
print(spark.catalog.listTables())
-----------------------------------------
'''Query Data'''

query = "FROM flights SELECT * LIMIT 10"

# Get the first 10 rows of flights
flights10 = spark.sql(query)

# Show the results
flights10.show()
----------------------------------
'''Spark to Pandas'''

# Don't change this query
query = "SELECT origin, dest, COUNT(*) as N FROM flights GROUP BY origin, dest"

# Run the query
flight_counts = spark.sql(query)

# Convert the results to a pandas DataFrame
pd_counts = flight_counts.toPandas()

# Print the head of pd_counts
print(pd_counts.head())

---------------------------------------
'''Pandas to Spark'''

# Create pd_temp
pd_temp = pd.DataFrame(np.random.random(10))

# Create spark_temp from pd_temp
spark_temp = spark.createDataFrame(pd_temp)

# Examine the tables in the catalog
print(spark.catalog.listTables())

# Add spark_temp to the catalog
spark_temp.createOrReplaceTempView("temp")

# Examine the tables in the catalog again
print(spark.catalog.listTables())
--------------------------------------------
'''read data directly'''

# Don't change this file path
file_path = "/usr/local/share/datasets/airports.csv"

# Read in the airports data
airports = spark.read.csv(file_path, header=True)

# Show the data
airports.show()
--------------------------------------
#CHAPTER 2 MANIPULATING DATA

# Create the DataFrame with flights table in the .catalog
flights = spark.table("flights")

# Show the head
print(flights.show())

# Add duration_hrs
flights = flights.withColumn("duration_hrs", flights.air_time/60)
-----------------------------------------------------------------
'''filtering '''
#spark session w/ spark dataframe = flights

# Filter flights with a SQL string
long_flights1 = flights.filter("distance > 1000")

# Filter flights with a boolean column
long_flights2 = flights.filter(flights.distance > 1000)

# Examine the data to check they're equal
print(long_flights1.show())
print(long_flights2.show())
-------------------------------------------------------------------
'''Selecting'''
# Select the first set of columns
selected1 = flights.select("tailnum", "origin", "dest")

# Select the second set of columns
temp = flights.select(flights.origin, flights.dest, flights.carrier)

# Define first filter
filterA = flights.origin == "SEA"

# Define second filter
filterB = flights.dest == "PDX"

# Filter the data, first by filterA then by filterB
selected2 = temp.filter(filterA).filter(filterB)
------------------------------------------------------
'''Selecting 2'''

# Define avg_speed using alias
avg_speed = (flights.distance/(flights.air_time/60)).alias("avg_speed")

# Select the correct columns
speed1 = flights.select("origin", "dest", "tailnum", avg_speed)

# Create the same table using a SQL expression 'as'
speed2 = flights.selectExpr("origin", "dest", "tailnum", "distance/(air_time/60) as avg_speed")

-------------------------------------------------------
'''aggregating
.groupBy()
.min() .max() .count() .avg() .sum()'''

# Find the shortest flight from PDX in terms of distance
flights.filter(flights.origin == "PDX").groupBy().min("distance").show()

# Find the longest flight from SEA in terms of duration
flights.filter(flights.origin == "SEA").groupBy().max("air_time").show()

----------------------------------------------------------
'''aggregating 2'''
# Average duration of Delta flights
flights.filter(flights.carrier == "DL").filter(flights.origin == "SEA").groupBy().avg("air_time").show()

# Total hours in the air
flights.withColumn("duration_hrs", flights.air_time/60).groupBy().sum("duration_hrs").show()

-----------------------------------------------------------
''' Grouping and Aggregating'''
# Group by tailnum
by_plane = flights.groupBy("tailnum")

# Number of flights each plane made
by_plane.count().show()

# Group by origin
by_origin = flights.groupBy("origin")

# Average duration of flights from PDX and SEA
by_origin.avg("air_time").show()
-----------------------------------------------------------
''' Grouping and Aggregating 2'''
# Import pyspark.sql.functions as F
import pyspark.sql.functions as F

# Group by month and dest
by_month_dest = flights.groupBy("month", "dest")

# Average departure delay by month and destination
by_month_dest.avg("dep_delay").show()

# Standard deviation
by_month_dest.agg(F.stddev("dep_delay")).show()
-----------------------------------------------------------
''' joining '''

# Examine the data
print(airports.show())

# Rename the faa column
airports = airports.withColumnRenamed("faa", "dest")

# Join the DataFrames
flights_with_airports = flights.join(airports, on="dest", how="leftouter")

# Examine the data again
print(flights_with_airports.show())


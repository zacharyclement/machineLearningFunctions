data = xrange(1, 10001)

ds = sc.parallelize(data, 8)

print ds.collect()

#1. DOWNLOAD EXTERNAL DATA
%sh curl -O 'https://raw.githubusercontent.com/bsullins/bensullins.com-freebies/master/CogsleyServices-SalesData-US.csv'
%sq #use sql query
%fs #look into dbfs

#2. READ AND CLEANSE DATA
#example 1
path = 'file:/databricks/driver/CogsleyServices-SalesData-US.csv'
data = sqlContext.read.format("csv")\
    .option("header", "true")\
    .option("inferSchema", "true")\
    .load(path)

  data.cache()
  data = data.dropna()

#example 2
dataPath = "/databricks-datasets/Rdatasets/data-001/csv/ggplot2/diamonds.csv"
diamonds = spark.read.format("csv")\
  .option("header","true")\
  .option("inferSchema", "true")\
  .load(dataPath)

display(diamonds)

df1 = diamonds.groupBy("cut", "color").avg("price") # a simple grouping

df2 = df1.join(diamonds, on='color', how='inner').select("`avg(price)`", "carat")

df2.cache()

df2.count()


#3. AGGREGATE AND CONVERT


#4 BUILD FEATURES AND LABELS FOR REGRESSION
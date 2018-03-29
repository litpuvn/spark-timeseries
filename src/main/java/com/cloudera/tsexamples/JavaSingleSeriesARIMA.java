package com.cloudera.tsexamples;

import com.cloudera.sparkts.models.ARIMA;
import com.cloudera.sparkts.models.ARIMAModel;
import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.mllib.linalg.Vectors;
import org.apache.spark.sql.SQLContext;

import java.util.List;

public class JavaSingleSeriesARIMA {
    public static void main(String[] args) {
        SparkConf conf = new SparkConf().setAppName("Spark-TS Ticker Example").setMaster("local");
        conf.set("spark.io.compression.codec", "org.apache.spark.io.LZ4CompressionCodec");
        JavaSparkContext context = new JavaSparkContext(conf);
        SQLContext sqlContext = new SQLContext(context);

        JavaRDD<Double> doubleJavaRDD = context.textFile("data/R_ARIMA_DataSet1.csv")
                .map(Double::parseDouble);
        List<Double> doubleList = doubleJavaRDD.collect();
        double[] doubleArray = doubleList.stream().mapToDouble(d -> d).toArray();
        org.apache.spark.mllib.linalg.Vector doubleVector = Vectors.dense(doubleArray);

        ARIMAModel model = ARIMA.autoFit(doubleVector, 1, 0, 1);

        // forecast next 20 values
        System.out.print("forecast: " + model.forecast(doubleVector, 20));

    }
}

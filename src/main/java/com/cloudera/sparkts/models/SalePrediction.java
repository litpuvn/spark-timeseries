package com.cloudera.sparkts.models;

import static org.apache.spark.sql.functions.*;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.LinkedList;
import java.util.List;

import org.apache.commons.lang.ArrayUtils;
import org.apache.spark.SparkConf;
import org.apache.spark.SparkContext;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.ml.Pipeline;
import org.apache.spark.ml.PipelineModel;
import org.apache.spark.ml.PipelineStage;
import org.apache.spark.ml.feature.Interaction;
import org.apache.spark.ml.feature.OneHotEncoder;
import org.apache.spark.ml.feature.StringIndexer;
import org.apache.spark.ml.feature.StringIndexerModel;
import org.apache.spark.ml.feature.VectorAssembler;
import org.apache.spark.ml.param.Param;
import org.apache.spark.ml.param.ParamMap;
import org.apache.spark.mllib.linalg.Matrix;
import org.apache.spark.mllib.linalg.SingularValueDecomposition;
import org.apache.spark.mllib.linalg.Vector;
import org.apache.spark.mllib.linalg.Vectors;
import org.apache.spark.mllib.linalg.distributed.RowMatrix;
import org.apache.spark.sql.*;
import org.apache.spark.sql.types.DataTypes;
import org.apache.spark.sql.types.Metadata;
import org.apache.spark.sql.types.StructField;
import org.apache.spark.sql.types.StructType;

import scala.Option;
import scala.reflect.ClassTag;
//import shapeless.newtype;
import breeze.linalg.DenseMatrix;
import breeze.linalg.DenseVector;
//import org.apache.spark.sql.Row;
//import org.apache.spark.mllib.linalg.DenseVector;
import org.apache.spark.sql.SparkSession;

public class SalePrediction {

    public static void main(String args[])
    {
//        System.setProperty("hadoop.home.dir", "C:/winutils");

        SparkConf conf = new SparkConf().setAppName("example").setMaster("local[*]");
        SparkContext sparkContext = new SparkContext(conf);
        SparkSession sparkSession = new SparkSession(sparkContext);

        SQLContext sqlContext = new SQLContext(sparkContext);

        Dataset<String> lines = sqlContext.read().textFile("data/sale-data.csv");
        Dataset<Row> data = sqlContext.read().csv("data/sale-prediction_data.csv");

        StructType schema = new StructType(new StructField[]{
                new StructField("ProductCategory", DataTypes.StringType, false, Metadata.empty()),
                new StructField("State", DataTypes.StringType, false, Metadata.empty())});

        Dataset<Row> df = sqlContext.createDataFrame(data.rdd(), schema);
        df = df.withColumn("test_index", monotonically_increasing_id());


        //df.show();
        StringIndexerModel indexer = new StringIndexer()
                .setInputCol("ProductCategory")
                .setOutputCol("categoryIndex")
                .fit(df);

        StringIndexerModel indexe = new StringIndexer()
                .setInputCol("State")
                .setOutputCol("stateIndex")
                .fit(df);

        Dataset<Row> indexed = indexer.transform(df);
        //indexed.show();

        Dataset<Row> indxed = indexe.transform(df);
        //indxed.show();

        DenseMatrix<Object> xreg = test(df, indexed, indxed, sparkSession);

        Dataset<Double> doubleData = lines.map(line->Double.parseDouble(line.toString()),
                Encoders.DOUBLE());

        List<Double> doubleList = doubleData.collectAsList();

        Double[] doubleArray = new Double[doubleList.size()];
        doubleArray = doubleList.toArray(doubleArray);

        double[] values = new double[doubleArray.length];
        for(int i = 0; i< doubleArray.length; i++)
        {
            values[i] = doubleArray[i];
        }

        Vector tsvector = Vectors.dense(values);
        System.out.println("Ts vector:" + tsvector.toString());

        double[] val = {1.0,1.0,1.0,1.0};
        org.apache.spark.mllib.linalg.DenseVector time = new org.apache.spark.mllib.linalg.DenseVector(val);
        Option<double[]> userInitParams = org.apache.spark.mllib.linalg.DenseVector.unapply(time);


        //			p, d, q
        ARIMAXModel model = ARIMAX.fitModel(1, 1, 0, tsvector, xreg, 1, false, false, userInitParams);

        //double[][] demo = model.differenceXreg(xreg);

        double[] coefficients = model.coefficients();
        for (double d : coefficients) {
            System.out.println(d+",");
        }
        System.out.println("coefficients  "+coefficients.length);
        System.out.println("model.xregMaxLag() : "+model.xregMaxLag() );
        System.out.println("p : "+model.p());
        System.out.println("d : "+model.d());
        System.out.println("q : "+model.q());

        DenseVector<Object> timeSeries = new DenseVector<Object>(tsvector.toArray());

        DenseVector<Object> forcast = model.forecast(timeSeries, xreg);

        System.out.println("Forcast:" +forcast);

    }

    public static DenseMatrix<Object> test(Dataset<Row> df,Dataset<Row> indexed,Dataset<Row> indxed,SparkSession sparkSession) {


        List<Column> columnList = new ArrayList<Column>();
        String[] indexedColNames = indexed.columns();
        for (String colmName : indexedColNames) {
            columnList.add(indexed.col(colmName));
        }
        Column[] arrColumn = new Column[columnList.size()];
        Dataset<Row> df1 = df.join(indexed, df.col("test_index").equalTo(indexed.col("test_index"))).select(columnList.toArray(arrColumn));
        //df1.show();

        Column[] arrColumn1 = new Column[2];
        arrColumn1[0] = df1.col("categoryIndex");
        arrColumn1[1] = indxed.col("stateIndex");

        Dataset<Row> df2 = df1.join(indxed, df1.col("test_index").equalTo(indxed.col("test_index"))).select(arrColumn1);
        df2.show();

        List<Row> lw = df2.collectAsList();

        int rowCount = lw.size();
        int colCount = lw.get(0).size();
        double[] arr =  new double[rowCount*colCount];

        int count = 0;
        for (int i = 0; i < colCount; i++) {
            for (int j = 0; j < rowCount; j++) {
                arr[count++] = lw.get(j).getDouble(i);
            }
        }

        System.out.println(arr);


        DenseMatrix<Object> xrg = new DenseMatrix<Object>(212, 2, arr);

        return xrg;





    }
}

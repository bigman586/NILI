package com.example.bigman586.plug;

import com.google.gson.JsonObject;

import java.util.List;

import okhttp3.ResponseBody;
import retrofit2.Call;
import retrofit2.http.Body;
import retrofit2.http.GET;
import retrofit2.http.Headers;
import retrofit2.http.POST;

public interface ConnectToServer {

    @Headers({"Content-type: application/json"})
    @GET("/getPrediction")
    Call<JsonObject> getPrediction();

    @Headers({"Accept:text/plain"})
    @GET("/getStatus")
    Call<String> getStatus();

    @Headers({"Accept:text/plain"})
    @GET("/getMean")
    Call<String> getMean();

    @Headers("Content-type: application/json")
    @GET("/getAllLabels")
    Call<List<String>> getAllLabels();

    @Headers("Content-type: application/json")
    @POST("/postCommand")
    Call<Void> postCommand(@Body JsonObject entry);

    @Headers("Content-type: application/json")
    @POST("/postLabel")
    Call<Void> postLabel(@Body JsonObject entry);
}
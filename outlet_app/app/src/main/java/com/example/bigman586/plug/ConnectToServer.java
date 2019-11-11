package com.example.bigman586.plug;

import com.google.gson.JsonObject;

import java.util.List;

import retrofit2.Call;
import retrofit2.http.Body;
import retrofit2.http.GET;
import retrofit2.http.Headers;
import retrofit2.http.POST;
import retrofit2.http.Path;

public interface ConnectToServer {

    @Headers("Content-type: application/json")
    @POST("/postCommand")
    Call<Void> postCommand(@Body JsonObject entry);

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
    @POST("/postLabel")
    Call<Void> postLabel(@Body JsonObject entry);

}

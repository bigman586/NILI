package com.example.bigman586.plug;

import android.os.Bundle;
import android.os.Handler;
import android.os.Looper;
import android.support.v4.app.Fragment;
import android.support.v4.app.FragmentActivity;
import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;
import android.widget.TextView;

import com.google.gson.JsonObject;
import com.jjoe64.graphview.series.DataPoint;

import retrofit2.Call;
import retrofit2.Callback;
import retrofit2.Response;

public class PredictFragment extends Fragment {
    // Store instance variables
    private String title;
    private int page;

    private TextView predictionView;

    public static PredictFragment newInstance(int page, String title) {

        PredictFragment fragmentFirst = new PredictFragment();
        Bundle args = new Bundle();

        args.putInt("1", page);
        args.putString("Predict", title);

        fragmentFirst.setArguments(args);
        return fragmentFirst;
    }

    // Store instance variables based on arguments passed
    @Override
    public void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        page = getArguments().getInt("someInt", 0);
        title = getArguments().getString("someTitle");
    }

    @Override
    public View onCreateView(LayoutInflater inflater, ViewGroup container,
                             Bundle savedInstanceState) {
        ViewGroup rootView2 = (ViewGroup) inflater.inflate(
                R.layout.activity_predict, container, false);

        predictionView = rootView2.findViewById(R.id.prediction);

//        final Runnable mTicker = new Runnable() {
//            public void run() {
//                //user interface updates on screen
//                getPrediction();
//                handler.postDelayed(mTicker, 1000);
//            }
//        };
//        handler.postDelayed(mTicker, 1000);
//        mTicker.run();

        getActivity().runOnUiThread(new Runnable() {
            public void run() {
                //user interface updates on screen
                getPrediction();

                Handler handler = new Handler();
                handler.postDelayed(this, 1000);
            }
        });
        return rootView2;
    }
    /**
     * predicted name of device plugged into outlet
     */
    public void getPrediction() {

        ConnectToServer service = RetrofitInstance.getRetrofitInstance().create(ConnectToServer.class);
        Call<JsonObject> call = service.getPrediction();

        call.enqueue(new Callback<JsonObject>() {

            MainPager.PagerAdapter adapter = new MainPager.PagerAdapter(((FragmentActivity) getContext()).getSupportFragmentManager());
            Fragment fragment = adapter.getItem(1);

            @Override
            public void onResponse(Call<JsonObject> call, Response<JsonObject> response) {
                    System.out.println(response.body());
                    JsonObject predictionJSON = response.body();

                    //changes the prediction text
                    assert predictionJSON != null;
                    String predictionString = String.format("%s", predictionJSON.get("prediction").toString());

                    predictionString = predictionString.replace("\"", "");
                    predictionView.setText(predictionString);
            }

            @Override
            public void onFailure(Call<JsonObject> call, Throwable t) {
                Utilities.errorToast(getString(R.string.error_no_connection), getContext());
            }

        });
    }
}
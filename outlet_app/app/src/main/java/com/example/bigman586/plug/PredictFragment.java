package com.example.bigman586.plug;

import android.content.Intent;
import android.os.Bundle;
import android.support.v4.app.Fragment;
import android.support.v7.app.AppCompatActivity;
import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;

public class PredictFragment extends Fragment {
    // Store instance variables
    private String title;
    private int page;

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

        return rootView2;
    }
}
package com.example.bigman586.plug;

import android.graphics.Color;
import android.graphics.Point;
import android.graphics.Typeface;
import android.os.Bundle;
import android.os.Handler;
import android.os.Looper;
import android.support.design.widget.TabLayout;
import android.support.v4.view.ViewPager;
import android.support.v7.app.ActionBar;
import android.support.v7.app.AppCompatActivity;
import android.view.Display;
import android.view.ViewGroup;
import android.widget.CompoundButton;
import android.widget.Switch;
import android.widget.TextView;
import android.widget.Toast;

import com.google.gson.JsonObject;
import com.jjoe64.graphview.GraphView;
import com.jjoe64.graphview.Viewport;
import com.jjoe64.graphview.series.DataPoint;
import com.jjoe64.graphview.series.LineGraphSeries;

import retrofit2.Call;
import retrofit2.Callback;
import retrofit2.Response;


public class MainActivity extends AppCompatActivity {
    private Switch toggle;
    private TextView on, off, currentMean;
    private String status;
    private double mean;
    private LineGraphSeries<DataPoint> series;
    private int lastX = 0;

    ViewPager viewPager;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        //removes actionbar from screen
        ActionBar actionBar = getSupportActionBar();

        assert actionBar != null;
        actionBar.hide();

        toggle = findViewById(R.id.toggle);

        viewPager = findViewById(R.id.viewpager);
        viewPager.setAdapter(new MainPager.PagerAdapter(getSupportFragmentManager()));

/*
        tabLayout= findViewById(R.id.tab_view);
        tabLayout.setupWithViewPager(viewPager);
*/

        on = findViewById(R.id.onText);
        off = findViewById(R.id.offText);

        currentMean = findViewById(R.id.currentMean);
        currentMean.setText("");

        mean = 0;

        toggle.setOnCheckedChangeListener(new CompoundButton.OnCheckedChangeListener() {
            public void onCheckedChanged(CompoundButton buttonView, boolean isChecked) {
                if (isChecked) {
                    status = "on";
                    sendCommand();

                    on.setTypeface(null, Typeface.BOLD);
                    on.setTextSize(20f);
                    on.setTextColor(Color.parseColor("#000000"));

                    off.setTypeface(null, Typeface.NORMAL);
                    off.setTextSize(17.5f);
                    off.setTextColor(Color.parseColor("#cecece"));

                } else {

                    status = "off";
                    sendCommand();

                    off.setTypeface(null, Typeface.BOLD);
                    off.setTextSize(20f);
                    off.setTextColor(Color.parseColor("#000000"));

                    on.setTypeface(null, Typeface.NORMAL);
                    on.setTextSize(17.5f);
                    on.setTextColor(Color.parseColor("#cecece"));
                }
            }
        });

        updateSwitch();

        // we get graph view instance
        GraphView graph = findViewById(R.id.graph);
        // data
        series = new LineGraphSeries<DataPoint>();
        graph.addSeries(series);

        // customize viewport
        Viewport viewport = graph.getViewport();
        viewport.setYAxisBoundsManual(true);
        viewport.setMinY(0);
        viewport.setMaxY(5);
        viewport.setScrollable(true);
        graph.getGridLabelRenderer().setVerticalAxisTitle("Current (A)");
        graph.getGridLabelRenderer().setHorizontalAxisTitle("Time (s)");
        updateInfo();

        Display display = getWindowManager().getDefaultDisplay();
        Point size = new Point();
        display.getSize(size);

        int height = size.y;

        // controls height of view pager
        ViewGroup.LayoutParams params = viewPager.getLayoutParams();
        params.height = height/3;
        viewPager.setLayoutParams(params);

        NetworkSniffTask test = new NetworkSniffTask(MainActivity.this.getApplicationContext());
        test.doInBackground();
    }

    private final int SECONDS = 1000;
    Handler handler;

    public void updateInfo() {
        handler = new Handler(Looper.getMainLooper());
        handler.postDelayed(new Runnable() {
            public void run() {
                handler = new Handler(Looper.getMainLooper());
                updateMean();
                series.appendData(new DataPoint(lastX++, mean), true, 20);     // this method will contain your almost-finished HTTP calls
                handler.postDelayed(this, SECONDS);
            }
        }, SECONDS);
    }

    /**
     * add random data to graph
     */
    private void updateMean() {
        ConnectToServer service = RetrofitInstance.getRetrofitInstance().create(ConnectToServer.class);

        Call<String> call = service.getMean();
        call.enqueue(new Callback<String>(){

            @Override
            public void onResponse(Call<String> call, Response<String> response) {
                String current = response.body();

                System.out.println(current);
                mean = Double.parseDouble((current));

                currentMean.setText(String.format("%s A", Utilities.round(mean, 5)));
            }

            @Override
            public void onFailure(Call<String> call, Throwable t) {
                Utilities.errorToast(getString(R.string.error_no_connection), MainActivity.this.getApplicationContext());
            }

        });
    }

    /**
     * updates button status based on outlet status
     */
    public void updateSwitch(){
        ConnectToServer service = RetrofitInstance.getRetrofitInstance().create(ConnectToServer.class);

        Call<String> call = service.getStatus();
        call.enqueue(new Callback<String>(){

            @Override
            public void onResponse(Call<String> call, Response<String> response) {
                String state = response.body();

                System.out.println(state);

                assert state != null;
                if (state.equals("on")){
                    status = "on";
                    toggle.setChecked(true);
                }
                else if(state.equals("off")){
                    status ="off";
                    toggle.setChecked(false);
                }
            }

            @Override
            public void onFailure(Call<String> call, Throwable t) {
                Utilities.errorToast(getString(R.string.error_no_connection), MainActivity.this.getApplicationContext());
            }

        });
    }

    /**
     * sends command to server
     */
    public void sendCommand(){

        JsonObject jsonObj = new JsonObject();
        jsonObj.addProperty("command", status);

        ConnectToServer service = RetrofitInstance.getRetrofitInstance().create(ConnectToServer.class);

        Call call = service.postCommand(jsonObj);
        call.enqueue(new Callback(){

            @Override
            public void onFailure(Call call, Throwable t) {
                Utilities.errorToast(getString(R.string.error_no_connection), MainActivity.this.getApplicationContext());
            }

            @Override
            public void onResponse(Call call, Response response) {

                if(response.isSuccessful()) {
                    Toast.makeText(MainActivity.this.getApplicationContext(),
                            ("Switched " + status + " the outlet"), Toast.LENGTH_SHORT).show();
                }
            }
        });
    }
}
package com.example.bigman586.plug;

import android.app.ProgressDialog;
import android.content.Context;
import android.graphics.Color;
import android.graphics.Typeface;
import android.os.Bundle;
import android.os.Handler;
import android.os.Looper;
import android.support.v7.app.ActionBar;
import android.support.v7.app.AppCompatActivity;
import android.text.TextUtils;
import android.view.View;
import android.view.inputmethod.InputMethodManager;
import android.widget.ArrayAdapter;
import android.widget.AutoCompleteTextView;
import android.widget.CompoundButton;
import android.widget.EditText;
import android.widget.ImageButton;
import android.widget.Switch;
import android.widget.TextView;
import android.widget.Toast;

import com.google.gson.JsonObject;

import java.math.BigDecimal;
import java.math.RoundingMode;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;

import retrofit2.Call;
import retrofit2.Callback;
import retrofit2.Response;

import com.jjoe64.graphview.GraphView;
import com.jjoe64.graphview.Viewport;
import com.jjoe64.graphview.series.DataPoint;
import com.jjoe64.graphview.series.LineGraphSeries;

/**
 * A screen that allows user to label data
 */
public class LabelActivity extends AppCompatActivity{

    // UI references.
    /*private EditText mPasswordView;
    private View mProgressView;
    private View mLoginFormView;*/
    private ImageButton submit;
    private AutoCompleteTextView labelField;
    private Switch toggle;
    private TextView on, off, currentDevice, currentMean;
    private ProgressDialog progressBar;
    private String entry;
    private EditText placeHolder;
    private String status;
    private double mean;

    private ArrayList<String> autoList;

    private static final Random RANDOM = new Random();
    private LineGraphSeries<DataPoint> series;
    private int lastX = 0;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_label);

        //creates progress bar
        progressBar = new ProgressDialog(LabelActivity.this);
        progressBar.setMessage("Loading....");

        //removes actionbar from screen
        ActionBar actionBar = getSupportActionBar();
        actionBar.hide();

        //Connect UI Elements to variables
        submit = (ImageButton) findViewById(R.id.send);
        labelField = (AutoCompleteTextView) findViewById(R.id.labelField);
        toggle = (Switch) findViewById(R.id.toggle);

        autoList = new ArrayList<String>();
        ArrayAdapter<String> adapter =
                new ArrayAdapter<>(LabelActivity.this,
                        android.R.layout.simple_dropdown_item_1line, autoList);

        labelField.setAdapter(adapter);

        placeHolder = (EditText) findViewById(R.id.placeHolder);

        on = (TextView) findViewById(R.id.onText);
        off = (TextView) findViewById(R.id.offText);

        currentDevice = (TextView) findViewById(R.id.currentDevice);
        currentMean = (TextView) findViewById(R.id.currentMean);

        currentDevice.setText("");
        currentMean.setText("");

        mean = 0;

        submit.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                submitLabel();
                hideKeyboard();
                labelField.setText("");
            }
        });

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

        fillLabelArray();

        // we get graph view instance
        GraphView graph = (GraphView) findViewById(R.id.graph);
        // data
        series = new LineGraphSeries<DataPoint>();
        graph.addSeries(series);
        // customize a little bit viewport
        Viewport viewport = graph.getViewport();
        viewport.setYAxisBoundsManual(true);
        viewport.setMinY(0);
        viewport.setMaxY(0.8);
        viewport.setScrollable(true);
        graph.getGridLabelRenderer().setVerticalAxisTitle("Current (A)");
        graph.getGridLabelRenderer().setHorizontalAxisTitle("Time (s)");
        updateInfo();
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

    // add random data to graph
    private void updateMean() {
        // here, we choose to display max 10 points on the viewport and we scroll to end
        ConnectToServer service = RetrofitInstance.getRetrofitInstance().create(ConnectToServer.class);

        Call<String> call = service.getMean();
        call.enqueue(new Callback<String>(){

            @Override
            public void onResponse(Call<String> call, Response<String> response) {
                String current = response.body();

                System.out.println(current);
                mean = Double.parseDouble((current));

                currentMean.setText(Utilities.round(mean, 5) + " A");
            }

            @Override
            public void onFailure(Call<String> call, Throwable t) {
                Utilities.errorToast(getString(R.string.error_no_connection), LabelActivity.this.getApplicationContext());
            }

        });

    }

    /**
     * Send label to server using retrofit
     */
    private void submitLabel() {
        this.entry = String.valueOf(labelField.getText());

        if(TextUtils.isEmpty(entry)){
            placeHolder.setError(getString(R.string.error_field_required));
            placeHolder.requestFocus();
            return;
        }

        if(!Utilities.isTextValid(entry)) {
            Toast.makeText(LabelActivity.this.getApplicationContext(),
                    (getString(R.string.error_wrong_label)), Toast.LENGTH_SHORT).show();
            return;
        }

        // shows progress bar
        progressBar.show();

        JsonObject jsonObj = new JsonObject();
        jsonObj.addProperty("label", entry);

        ConnectToServer service = RetrofitInstance.getRetrofitInstance().create(ConnectToServer.class);

        Call call = service.postLabel(jsonObj);
        call.enqueue(new Callback(){

            @Override
            public void onFailure(Call call, Throwable t) {

                Utilities.errorToast(getString(R.string.error_no_connection), LabelActivity.this.getApplicationContext());
                Utilities.errorToast(getString(R.string.error_no_send), LabelActivity.this.getApplicationContext());
            }

            @Override
            public void onResponse(Call call, Response response) {

                String article = "";

                if(entry.charAt(0) == 'a' || entry.charAt(0) == 'e' || entry.charAt(0) == 'i'
                        || entry.charAt(0) == 'o' || entry.charAt(0) == 'u'){
                    article = "An";
                }
                else{
                    article = "A";
                }

                if(response.isSuccessful()) {
                    Toast.makeText(LabelActivity.this.getApplicationContext(),
                            (article + " " + entry + " is plugged into the server"), Toast.LENGTH_SHORT).show();
                }
            }
        });

        progressBar.hide();
        currentDevice.setText(entry);
    }

    /**
     * @return array of all Labels stored in database
     */
    public void fillLabelArray(){

        ConnectToServer service = RetrofitInstance.getRetrofitInstance().create(ConnectToServer.class);

        Call<List<String>> call = service.getAllLabels();
        call.enqueue(new Callback<List<String>>(){

            @Override
            public void onFailure(Call<List<String>> call, Throwable t) {

                Utilities.errorToast(getString(R.string.error_no_connection), LabelActivity.this.getApplicationContext());
                Utilities.errorToast(getString(R.string.error_no_send), LabelActivity.this.getApplicationContext());
                return;

            }

            @Override
            public void onResponse(Call<List<String>> call, Response<List<String>> response) {
                List<String> list = response.body();
                for (String item :list) {

                    System.out.println(item);
                    autoList.add(item);
                }
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
                Utilities.errorToast(getString(R.string.error_no_connection), LabelActivity.this.getApplicationContext());
            }

        });
    }

    public void sendCommand(){

        JsonObject jsonObj = new JsonObject();
        jsonObj.addProperty("command", status);

        ConnectToServer service = RetrofitInstance.getRetrofitInstance().create(ConnectToServer.class);

        Call call = service.postCommand(jsonObj);
        call.enqueue(new Callback(){

            @Override
            public void onFailure(Call call, Throwable t) {
                Utilities.errorToast(getString(R.string.error_no_connection), LabelActivity.this.getApplicationContext());
            }

            @Override
            public void onResponse(Call call, Response response) {

                if(response.isSuccessful()) {
                    Toast.makeText(LabelActivity.this.getApplicationContext(),
                            ("Switched " + status + " the outlet"), Toast.LENGTH_SHORT).show();
                }
            }
        });
    }

    /**
     * Closes software keyboard
     */
    public void hideKeyboard() {
        View view = this.getCurrentFocus();
        if (view != null) {
            InputMethodManager imm = (InputMethodManager) getSystemService(Context.INPUT_METHOD_SERVICE);
            imm.hideSoftInputFromWindow(view.getWindowToken(), 0);
        }
    }




}





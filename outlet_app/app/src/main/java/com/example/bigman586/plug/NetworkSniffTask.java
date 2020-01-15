package com.example.bigman586.plug;

import android.content.Context;
import android.net.ConnectivityManager;
import android.net.NetworkInfo;
import android.net.wifi.WifiInfo;
import android.net.wifi.WifiManager;
import android.os.AsyncTask;
import android.text.format.Formatter;
import android.util.Log;

import java.lang.ref.WeakReference;
import java.net.InetAddress;

public class NetworkSniffTask extends AsyncTask<Void, Void, Void> {
    private static final String TAG = "NetworkSniffTask";
    private WeakReference<Context> mContextRef;

    NetworkSniffTask(Context context) {
        mContextRef = new WeakReference<>(context);
    }

    @Override
    protected Void doInBackground(Void... voids) {
        System.out.println("Let's sniff the network");
        try {
            Context context = mContextRef.get();
            if (context != null) {
                ConnectivityManager cm = (ConnectivityManager) context.getSystemService(Context.CONNECTIVITY_SERVICE);
                NetworkInfo activeNetwork = cm.getActiveNetworkInfo();

                WifiManager wm = (WifiManager) context.getApplicationContext().getSystemService(Context.WIFI_SERVICE);
                WifiInfo connectionInfo = wm.getConnectionInfo();

                int ipAddress = connectionInfo.getIpAddress();
                String ipString = Formatter.formatIpAddress(ipAddress);

                System.out.println("activeNetwork: " + String.valueOf(activeNetwork));
                System.out.println("ipString: " + String.valueOf(ipString));

                String prefix = ipString.substring(0, ipString.lastIndexOf(".") + 1);
                System.out.println("prefix: " + prefix);

                for (int i = 0; i < 255; i++) {
                    String testIp = prefix + String.valueOf(i);
                    InetAddress name = InetAddress.getByName(testIp);
                    String hostName = name.getCanonicalHostName();

                    if (name.isReachable(1000))
                        System.out.println("Host:" + hostName);
                }
            }
        } catch (Throwable t) {
            System.out.print("Well that's not good.");
        }
        return null;
    }
}


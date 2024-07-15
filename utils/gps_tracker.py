import gpxpy
import gpxpy.gpx
import random
import time
from datetime import datetime, timedelta

# Create a new GPX file
gpx = gpxpy.gpx.GPX()

# Add a new track to the GPX file
gpx_track = gpxpy.gpx.GPXTrack()
gpx.tracks.append(gpx_track)

# Add a new segment to the track
gpx_segment = gpxpy.gpx.GPXTrackSegment()
gpx_track.segments.append(gpx_segment)


# Generate simulated GPS points
start_time = datetime(2024, 7, 8, 12, 0, 0)
for i in range(100):
    latitude = 37.7749 + random.uniform(-0.001, 0.001)
    longitude = -122.4194 + random.uniform(-0.001, 0.001)
    altitude = 10 + random.uniform(-1, 1)
    speed = 10 + random.uniform(-2, 2)  # Speed in m/s
    time_stamp = start_time + timedelta(seconds=i * 0.1)

    # Add the GPS point to the GPX segment
    point = gpxpy.gpx.GPXTrackPoint(latitude, longitude, elevation=altitude, time=time_stamp, speed=speed)
    gpx_segment.points.append(point)

    # Wait for a short time before the next data point
    time.sleep(0.1)





# Print the generated GPX data
print(gpx.to_xml())

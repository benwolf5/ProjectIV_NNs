path = fullfile(toolboxdir('lidar'),'lidardata', ...
    'las','aerialLidarData.laz');
lasReader = lasFileReader('NZ2741_P_7312_20110131_20110131.laz');
ptCloud = readPointCloud(lasReader);
figure
pcshow(ptCloud.Location)

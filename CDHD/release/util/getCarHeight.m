function height = getCarHeight(record, type)
bbox = record.bbox;
height = bbox(4) - bbox(2) + 1;
end

package tensorboard

import "testing"

func TestReaderRecognisesRetainedLegacyFieldSixHistogram(t *testing.T) {
	value := appendString(nil, 1, "model/parameters")
	value = appendBytes(value, 6, histogramProto([]float64{-1, 0, 1}))
	scalar, histogram, err := decodeSummaryValue(value, 100, "retained-production-events")
	if err != nil {
		t.Fatal(err)
	}
	if scalar != nil || histogram == nil || histogram.Tag != "model/parameters" || histogram.Step != 100 || histogram.Count != 3 {
		t.Fatalf("legacy field-six decode = scalar %#v histogram %#v", scalar, histogram)
	}
}

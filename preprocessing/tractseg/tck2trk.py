import argparse
import nibabel as nib

def convert_tck_to_trk(tck_filename, nii_filename, trk_filename):
    # Load the .tck file using nibabel's streamlines API
    tck_file = nib.streamlines.load(tck_filename)
    # Ensure the correct reference image in nii format is used to obtain header information.
    reference_image = nib.load(nii_filename) 
    header = nib.streamlines.trk.TrkFile.create_empty_header()
    header['dimensions'] = reference_image.header.get_data_shape()[:3]
    header['voxel_sizes'] = reference_image.header.get_zooms()[:3]
    header['voxel_to_rasmm'] = reference_image.affine
    # Save the streamlines 
    tractogram = nib.streamlines.Tractogram(streamlines=tck_file.streamlines, affine_to_rasmm=reference_image.affine) 
    # Create a .trk file and save it
    trk_file = nib.streamlines.trk.TrkFile(tractogram, header=header)
    nib.streamlines.save(trk_file, trk_filename)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert a .tck file to a .trk file.")
    parser.add_argument("tck_filename", help="Path to the input .tck file.")
    parser.add_argument("nii_filename", help="Path to the reference .nii file.")
    parser.add_argument("trk_filename", help="Path to the output .trk file.")
    args = parser.parse_args()
    convert_tck_to_trk(args.tck_filename, args.nii_filename, args.trk_filename)

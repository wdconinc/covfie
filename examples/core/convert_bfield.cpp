/*
 * SPDX-PackageName: "covfie, a part of the ACTS project"
 * SPDX-FileCopyrightText: 2022 CERN
 * SPDX-License-Identifier: MPL-2.0
 */

#include <fstream>
#include <iostream>

#include <boost/log/trivial.hpp>
#include <boost/program_options.hpp>

#include <covfie/core/algebra/affine.hpp>
#include <covfie/core/backend/primitive/array.hpp>
#include <covfie/core/backend/transformer/affine.hpp>
#include <covfie/core/backend/transformer/nearest_neighbour.hpp>
#include <covfie/core/backend/transformer/strided.hpp>
#include <covfie/core/field.hpp>
#include <covfie/core/parameter_pack.hpp>

void parse_opts(
    int argc, char * argv[], boost::program_options::variables_map & vm
)
{
    boost::program_options::options_description opts("general options");

    opts.add_options()("help", "produce help message")(
        "input,i",
        boost::program_options::value<std::string>()->required(),
        "input magnetic field to read"
    )("output,o",
      boost::program_options::value<std::string>()->required(),
      "output magnetic field to write"
    )("scale,s",
      boost::program_options::value<float>()->default_value(0.000299792458f),
      "unit conversion scaling factor (default: 0.000299792458)");

    boost::program_options::parsed_options parsed =
        boost::program_options::command_line_parser(argc, argv)
            .options(opts)
            .run();

    boost::program_options::store(parsed, vm);

    if (vm.count("help")) {
        std::cout << opts << std::endl;
        std::exit(0);
    }

    try {
        boost::program_options::notify(vm);
        float scale_factor = vm["scale"].as<float>();
        if (scale_factor <= 0.0f) {
            BOOST_LOG_TRIVIAL(fatal) << "Invalid scale factor: " << scale_factor
                                     << ". It must be positive!";
            std::exit(1);
        }
    } catch (boost::program_options::required_option & e) {
        BOOST_LOG_TRIVIAL(error) << e.what();
        std::exit(1);
    }
}

using field_t = covfie::field<covfie::backend::affine<
    covfie::backend::nearest_neighbour<covfie::backend::strided<
        covfie::vector::size2,
        covfie::backend::array<covfie::vector::float2>>>>>;

field_t read_bfield(const std::string & fn, const float scale_factor)
{
    std::ifstream f;

    float minr = std::numeric_limits<float>::max();
    float maxr = std::numeric_limits<float>::lowest();
    float minz = std::numeric_limits<float>::max();
    float maxz = std::numeric_limits<float>::lowest();
    double spacing_r = 0.0, spacing_z = 0.0;

    {
        BOOST_LOG_TRIVIAL(info)
            << "Opening magnetic field to compute field limits";

        f.open(fn);

        if (!f.good()) {
            BOOST_LOG_TRIVIAL(fatal)
                << "Failed to open input file " << fn << "!";
            std::exit(1);
        }

        std::string line;

        float rp, zp;
        float Br, Bz;

        std::size_t n_lines = 0;

        BOOST_LOG_TRIVIAL(info)
            << "Iterating over lines in the magnetic field file";

        bool r_updated = false, z_updated = false;

        /*
         * Read every line, and update our current minima and maxima
         * appropriately.
         */
        while (f >> rp >> zp >> Br >> Bz) {
            if (n_lines == 0) {
                // Initialize sample spacing with the first coordinate values
                spacing_r = rp;
                spacing_z = zp;
            } else {
                // Update sample spacing per coordinate once when they change
                if (!r_updated && rp != spacing_r) {
                    spacing_r = std::abs(rp - spacing_r);
                    r_updated = true;
                }
                if (!z_updated && zp != spacing_z) {
                    spacing_z = std::abs(zp - spacing_z);
                    z_updated = true;
                }
            }

            minr = std::min(minr, rp);
            maxr = std::max(maxr, rp);

            minz = std::min(minz, zp);
            maxz = std::max(maxz, zp);

            ++n_lines;
        }

        BOOST_LOG_TRIVIAL(info)
            << "Read " << n_lines << " lines of magnetic field data";

        BOOST_LOG_TRIVIAL(info) << "Closing magnetic field file";

        f.close();
    }

    BOOST_LOG_TRIVIAL(info)
        << "Field dimensions in r = [" << minr << ", " << maxr << "]";
    BOOST_LOG_TRIVIAL(info)
        << "Field dimensions in z = [" << minz << ", " << maxz << "]";

    BOOST_LOG_TRIVIAL(info) << "Computed sample spacing:";
    BOOST_LOG_TRIVIAL(info) << "  r-spacing: " << spacing_r;
    BOOST_LOG_TRIVIAL(info) << "  z-spacing: " << spacing_z;

    if (spacing_r == 0.0 || spacing_z == 0.0) {
        BOOST_LOG_TRIVIAL(fatal)
            << "Sample spacing is 0 in one dimension! Error in calculating the "
               "sample spacing from file";
        std::exit(1);
    }

    /*
     * Now that we have the limits of our field, compute the size in each
     * dimension.
     */
    std::size_t sr =
        static_cast<std::size_t>(std::lround((maxr - minr) / spacing_r)) + 1;
    std::size_t sz =
        static_cast<std::size_t>(std::lround((maxz - minz) / spacing_z)) + 1;

    BOOST_LOG_TRIVIAL(info)
        << "Magnetic field size is " << sr << "x" << sz;

    BOOST_LOG_TRIVIAL(info) << "Constructing matching vector field...";

    covfie::algebra::affine<2> translation =
        covfie::algebra::affine<2>::translation(0.0f, -minz);
    covfie::algebra::affine<2> scaling = covfie::algebra::affine<2>::scaling(
        static_cast<float>(sr - 1) / (maxr - minr),
        static_cast<float>(sz - 1) / (maxz - minz)
    );

    field_t field(covfie::make_parameter_pack(
        field_t::backend_t::configuration_t(scaling * translation),
        field_t::backend_t::backend_t::configuration_t{},
        field_t::backend_t::backend_t::backend_t::configuration_t{sr, sz}
    ));
    field_t::view_t fv(field);

    {
        BOOST_LOG_TRIVIAL(info) << "Re-opening magnetic field to gather data";

        f.open(fn);

        if (!f.good()) {
            BOOST_LOG_TRIVIAL(fatal)
                << "Failed to open input file " << fn << "!";
            std::exit(1);
        }

        std::string line;

        float rp, zp;
        float Br, Bz;

        std::size_t n_lines = 0;

        BOOST_LOG_TRIVIAL(info)
            << "Iterating over lines in the magnetic field file";

        /*
         * Read every line, and update our current minima and maxima
         * appropriately.
         */
        while (f >> rp >> zp >> Br >> Bz) {
            field_t::view_t::output_t & p = fv.at(rp, zp);

            p[0] = Br * scale_factor;
            p[1] = Bz * scale_factor;

            ++n_lines;
        }

        BOOST_LOG_TRIVIAL(info)
            << "Read " << n_lines << " lines of magnetic field data";

        BOOST_LOG_TRIVIAL(info) << "Closing magnetic field file";

        f.close();
    }

    return field;
}

int main(int argc, char ** argv)
{
    boost::program_options::variables_map vm;
    parse_opts(argc, argv, vm);

    BOOST_LOG_TRIVIAL(info) << "Welcome to the covfie magnetic field renderer!";
    BOOST_LOG_TRIVIAL(info) << "Using magnetic field file \""
                            << vm["input"].as<std::string>() << "\"";
    BOOST_LOG_TRIVIAL(info) << "Starting read of input file...";

    field_t fb =
        read_bfield(vm["input"].as<std::string>(), vm["scale"].as<float>());

    BOOST_LOG_TRIVIAL(info) << "Writing magnetic field to file \""
                            << vm["output"].as<std::string>() << "\"...";

    std::ofstream fs(vm["output"].as<std::string>(), std::ofstream::binary);

    if (!fs.good()) {
        BOOST_LOG_TRIVIAL(fatal) << "Failed to open output file "
                                 << vm["output"].as<std::string>() << "!";
        std::exit(1);
    }

    fb.dump(fs);

    fs.close();

    BOOST_LOG_TRIVIAL(info) << "Rendering complete, goodbye!";

    return 0;
}

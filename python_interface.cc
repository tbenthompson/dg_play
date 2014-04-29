#include <Python.h>
#if PY_VERSION_HEX >= 0x03000000
    // The code should never get here, but if it does,
    // we should quit, because the conflicting interpreter and include
    // will cause weird problems.
    #error "Python 3?!" 
#endif
#include <boost/python.hpp>
#include <boost/python/suite/indexing/indexing_suite.hpp>
#include <boost/python/suite/indexing/vector_indexing_suite.hpp>
#include <boost/array.hpp>

#include <deal.II/base/point.h>
#include <deal.II/base/function.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/lac/parallel_vector.h>
#include <deal.II/distributed/solution_transfer.h>

#include "dg.h"
#include "control.h"


/* Note: error: ‘dealii::DoFHandler<dim, spacedim>::DoFHandler(const dealii::
 * DoFHandler<dim, spacedim>&) [with int dim = 2; int spacedim = 2]’ is private
 * means that a class' copy constructor is private and we should use
 * boost::noncopyable
 */
BOOST_PYTHON_MODULE(viscosaur)
{
    using namespace boost::python;
    namespace vc = viscosaur;

    class_<vc::Vc>("Vc", init<list, dict>())
        .def("get_rank", &vc::Vc::get_rank);
    
    class_<Step12<2>, boost::noncopyable >("DGMethod", init<bp::dict>())
        .def("run", &Step12<2>::run);
}


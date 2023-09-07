// clang-format off
/* ----------------------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   https://www.lammps.org/, Sandia National Laboratories
   LAMMPS development team: developers@lammps.org

   Copyright (2003) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under
   the GNU General Public License.

   See the README file in the top-level LAMMPS directory.
------------------------------------------------------------------------- */

#include "fix_rigid_shell.h"
#include "atom.h"
#include "atom_vec_ellipsoid.h"
#include "atom_vec_line.h"
#include "atom_vec_tri.h"
#include "comm.h"
#include "domain.h"
#include "error.h"
#include "force.h"
#include "group.h"
#include "hashlittle.h"
#include "input.h"
#include "integrate.h"
#include "math_const.h"
#include "math_eigen.h"
#include "math_extra.h"
#include "memory.h"
#include "modify.h"
#include "molecule.h"
#include "neighbor.h"
#include "neigh_list.h"
#include "library.h"
#include "random_mars.h"
#include "respa.h"
#include "rigid_const.h"
#include "tokenizer.h"
#include "update.h"
#include "variable.h"

#include <cmath>
#include <cstring>
#include <map>
#include <utility>
#include <cstdlib>
#include <iostream>

using namespace LAMMPS_NS;
using namespace FixConst;
using namespace MathConst;
using namespace RigidConst;

#define RVOUS 1   // 0 for irregular, 1 for all2all

/* ---------------------------------------------------------------------- */

FixRigidShell::FixRigidShell(LAMMPS *lmp, int narg, char **arg) :
  Fix(lmp, narg, arg), step_respa(nullptr),
  inpfile(nullptr), body(nullptr), bodyown(nullptr), bodytag(nullptr), atom2body(nullptr), angle2body(nullptr), vertexdata(nullptr), list(nullptr),
  xcmimage(nullptr), displace(nullptr), eflags(nullptr), orient(nullptr), dorient(nullptr),
  avec_ellipsoid(nullptr), avec_line(nullptr), avec_tri(nullptr), counts(nullptr),
  itensor(nullptr), mass_body(nullptr), langextra(nullptr), random(nullptr),
  id_dilate(nullptr), id_gravity(nullptr), onemols(nullptr)
{
  int i;

  scalar_flag = 1;
  extscalar = 0;
  global_freq = 1;
  time_integrate = 1;
  rigid_flag = 1;
  virial_global_flag = virial_peratom_flag = 1;
  thermo_virial = 1;
  create_attribute = 1;
  dof_flag = 1;
  enforce2d_flag = 1;
  stores_ids = 1;
  centroidstressflag = CENTROID_AVAIL;

  MPI_Comm_rank(world,&me);
  MPI_Comm_size(world,&nprocs);

  // perform initial allocation of atom-based arrays
  // register with Atom class

  extended = orientflag = dorientflag = customflag = 0;
  bodyown = nullptr;
  bodytag = nullptr;
  atom2body = nullptr;
  angle2body = nullptr;
  vertexdata = nullptr;
  xcmimage = nullptr;
  displace = nullptr;
  eflags = nullptr;
  orient = nullptr;
  dorient = nullptr;
  FixRigidShell::grow_arrays(atom->nmax);
  atom->add_callback(Atom::GROW);

  // parse args for rigid body specification

  int *mask = atom->mask;
  tagint *bodyID = nullptr;
  int nlocal = atom->nlocal;


  if (narg < 6) error->all(FLERR,"1 Illegal fix rigid/shell command");
  
  hstr = mustr = nullptr;
  hstyle = mustyle = CONSTANT;


  if (utils::strmatch(arg[3],"^v_")) {
    hstr = utils::strdup(arg[3]+2);
    hstyle = EQUAL;
  } else {
    pf = utils::numeric(FLERR,arg[3],false,lmp);
    // Conver from pressure units to force/distance^2
    pf /= force->nktv2p;
    hstyle = CONSTANT;
  }
  if (utils::strmatch(arg[4],"^v_")) {
    mustr = utils::strdup(arg[4]+2);
    mustyle = EQUAL;
  } else {
    mu = utils::numeric(FLERR,arg[4],false,lmp);
    mustyle = CONSTANT;
  }

  if (strcmp(arg[5],"molecule") == 0) {
    if (atom->molecule_flag == 0)
      error->all(FLERR,"Fix rigid/shell requires atom attribute molecule");
    bodyID = atom->molecule;

  } else if (strcmp(arg[5],"custom") == 0) {
    if (narg < 7) error->all(FLERR,"2 Illegal fix rigid/shell command");
      bodyID = new tagint[nlocal];
      customflag = 1;

      // determine whether atom-style variable or atom property is used

      if (utils::strmatch(arg[6],"^i_")) {
        int is_double,cols;
        int custom_index = atom->find_custom(arg[6]+2,is_double,cols);
        if (custom_index == -1)
          error->all(FLERR,"Fix rigid/shell custom requires previously defined property/atom");
        else if (is_double || cols)
          error->all(FLERR,"Fix rigid/shell custom requires integer-valued property/atom vector");
        int minval = INT_MAX;
        int *value = atom->ivector[custom_index];
        for (i = 0; i < nlocal; i++)
          if (mask[i] & groupbit) minval = MIN(minval,value[i]);
        int vmin = minval;
        MPI_Allreduce(&vmin,&minval,1,MPI_INT,MPI_MIN,world);

        for (i = 0; i < nlocal; i++)
          if (mask[i] & groupbit)
            bodyID[i] = (tagint)(value[i] - minval + 1);
          else bodyID[i] = 0;

      } else if (utils::strmatch(arg[6],"^v_")) {
        int ivariable = input->variable->find(arg[6]+2);
        if (ivariable < 0)
          error->all(FLERR,"Variable {} for fix rigid/shell custom does not exist", arg[6]+2);
        if (input->variable->atomstyle(ivariable) == 0)
          error->all(FLERR,"Fix rigid/shell custom variable {} is not atom-style variable",
                     arg[6]+2);
        auto value = new double[nlocal];
        input->variable->compute_atom(ivariable,0,value,1,0);
        int minval = INT_MAX;
        for (i = 0; i < nlocal; i++)
          if (mask[i] & groupbit) minval = MIN(minval,(int)value[i]);
        int vmin = minval;
        MPI_Allreduce(&vmin,&minval,1,MPI_INT,MPI_MIN,world);

        for (i = 0; i < nlocal; i++)
          if (mask[i] & groupbit)
            bodyID[i] = (tagint)((tagint)value[i] - minval + 1);
          else bodyID[0] = 0;
        delete[] value;
      } else error->all(FLERR,"Unsupported fix rigid custom property");
  } else error->all(FLERR,"3 Illegal fix rigid/shell command");

  if (atom->map_style == Atom::MAP_NONE)
    error->all(FLERR,"Fix rigid/shell requires an atom map, see atom_modify");

  // maxmol = largest bodyID #

  maxmol = -1;
  for (i = 0; i < nlocal; i++)
    if (mask[i] & groupbit) maxmol = MAX(maxmol,bodyID[i]);

  tagint itmp;
  MPI_Allreduce(&maxmol,&itmp,1,MPI_LMP_TAGINT,MPI_MAX,world);
  maxmol = itmp;

  // number of linear molecules is counted later
  nlinear = 0;

  // parse optional args

  int seed;
  langflag = 0;
  inpfile = nullptr;
  onemols = nullptr;
  reinitflag = 1;

  tstat_flag = 0;
  pstat_flag = 0;
  allremap = 1;
  id_dilate = nullptr;
  t_chain = 10;
  t_iter = 1;
  t_order = 3;
  p_chain = 10;

  pcouple = NONE;
  pstyle = ANISO;

  for (i = 0; i < 3; i++) {
    p_start[i] = p_stop[i] = p_period[i] = 0.0;
    p_flag[i] = 0;
  }

  int iarg = 6;
  if (customflag) ++iarg;

  while (iarg < narg) {
    if (strcmp(arg[iarg],"langevin") == 0) {
      if (iarg+5 > narg) error->all(FLERR,"4 Illegal fix rigid/shell command");
      if ((strcmp(style,"rigid/shell") != 0) &&
          (strcmp(style,"rigid/nve/shell") != 0) &&
          (strcmp(style,"rigid/nph/shell") != 0))
        error->all(FLERR,"5 Illegal fix rigid/shell command");
      langflag = 1;
      t_start = utils::numeric(FLERR,arg[iarg+1],false,lmp);
      t_stop = utils::numeric(FLERR,arg[iarg+2],false,lmp);
      t_period = utils::numeric(FLERR,arg[iarg+3],false,lmp);
      seed = utils::inumeric(FLERR,arg[iarg+4],false,lmp);
      if (t_period <= 0.0)
        error->all(FLERR,"Fix rigid/shell langevin period must be > 0.0");
      if (seed <= 0) error->all(FLERR,"6 Illegal fix rigid/shell command");
      iarg += 5;

    } else if (strcmp(arg[iarg],"infile") == 0) {
      if (iarg+2 > narg) error->all(FLERR,"7 Illegal fix rigid/shell command");
      delete[] inpfile;
      inpfile = utils::strdup(arg[iarg+1]);
      restart_file = 1;
      reinitflag = 0;
      iarg += 2;

    } else if (strcmp(arg[iarg],"reinit") == 0) {
      if (iarg+2 > narg) error->all(FLERR,"Illegal fix rigid/shell command");
      reinitflag = utils::logical(FLERR,arg[iarg+1],false,lmp);
      iarg += 2;

    } else if (strcmp(arg[iarg],"mol") == 0) {
      if (iarg+2 > narg) error->all(FLERR,"Illegal fix rigid/shell command");
      int imol = atom->find_molecule(arg[iarg+1]);
      if (imol == -1) error->all(FLERR,"Molecule template ID for fix rigid/shell does not exist");
      onemols = &atom->molecules[imol];
      nmol = onemols[0]->nset;
      restart_file = 1;
      iarg += 2;

    } else if (strcmp(arg[iarg],"temp") == 0) {
      if (iarg+4 > narg) error->all(FLERR,"Illegal fix rigid/shell command");
      if (!utils::strmatch(style,"^rigid/n.t/shell"))
        error->all(FLERR,"Illegal fix rigid command");
      tstat_flag = 1;
      t_start = utils::numeric(FLERR,arg[iarg+1],false,lmp);
      t_stop = utils::numeric(FLERR,arg[iarg+2],false,lmp);
      t_period = utils::numeric(FLERR,arg[iarg+3],false,lmp);
      iarg += 4;

    } else if (strcmp(arg[iarg],"iso") == 0) {
      if (iarg+4 > narg) error->all(FLERR,"Illegal fix rigid/shell command");
      if (!utils::strmatch(style,"^rigid/np./shell"))
        error->all(FLERR,"Illegal fix rigid/shell command");
      pcouple = XYZ;
      p_start[0] = p_start[1] = p_start[2] = utils::numeric(FLERR,arg[iarg+1],false,lmp);
      p_stop[0] = p_stop[1] = p_stop[2] = utils::numeric(FLERR,arg[iarg+2],false,lmp);
      p_period[0] = p_period[1] = p_period[2] =
        utils::numeric(FLERR,arg[iarg+3],false,lmp);
      p_flag[0] = p_flag[1] = p_flag[2] = 1;
      if (domain->dimension == 2) {
              p_start[2] = p_stop[2] = p_period[2] = 0.0;
        p_flag[2] = 0;
      }
      iarg += 4;

    } else if (strcmp(arg[iarg],"aniso") == 0) {
      if (iarg+4 > narg) error->all(FLERR,"Illegal fix rigid/shell command");
      if (!utils::strmatch(style,"^rigid/np./shell"))
        error->all(FLERR,"Illegal fix rigid/shell command");
      p_start[0] = p_start[1] = p_start[2] = utils::numeric(FLERR,arg[iarg+1],false,lmp);
      p_stop[0] = p_stop[1] = p_stop[2] = utils::numeric(FLERR,arg[iarg+2],false,lmp);
      p_period[0] = p_period[1] = p_period[2] =
        utils::numeric(FLERR,arg[iarg+3],false,lmp);
      p_flag[0] = p_flag[1] = p_flag[2] = 1;
      if (domain->dimension == 2) {
        p_start[2] = p_stop[2] = p_period[2] = 0.0;
              p_flag[2] = 0;
      }
      iarg += 4;

    } else if (strcmp(arg[iarg],"x") == 0) {
      if (iarg+4 > narg) error->all(FLERR,"Illegal fix rigid/shell command");
      if (!utils::strmatch(style,"^rigid/np./shell"))
        error->all(FLERR,"Illegal fix rigid/shell command");
      p_start[0] = utils::numeric(FLERR,arg[iarg+1],false,lmp);
      p_stop[0] = utils::numeric(FLERR,arg[iarg+2],false,lmp);
      p_period[0] = utils::numeric(FLERR,arg[iarg+3],false,lmp);
      p_flag[0] = 1;
      iarg += 4;

    } else if (strcmp(arg[iarg],"y") == 0) {
      if (iarg+4 > narg) error->all(FLERR,"Illegal fix rigid/shell command");
      if (!utils::strmatch(style,"^rigid/np./shell"))
        error->all(FLERR,"Illegal fix rigid/shell command");
      p_start[1] = utils::numeric(FLERR,arg[iarg+1],false,lmp);
      p_stop[1] = utils::numeric(FLERR,arg[iarg+2],false,lmp);
      p_period[1] = utils::numeric(FLERR,arg[iarg+3],false,lmp);
      p_flag[1] = 1;
      iarg += 4;

    } else if (strcmp(arg[iarg],"z") == 0) {
      if (iarg+4 > narg) error->all(FLERR,"Illegal fix rigid/shell command");
      if (!utils::strmatch(style,"^rigid/np./shell"))
        error->all(FLERR,"Illegal fix rigid/shell command");
      p_start[2] = utils::numeric(FLERR,arg[iarg+1],false,lmp);
      p_stop[2] = utils::numeric(FLERR,arg[iarg+2],false,lmp);
      p_period[2] = utils::numeric(FLERR,arg[iarg+3],false,lmp);
      p_flag[2] = 1;
      iarg += 4;

    } else if (strcmp(arg[iarg],"couple") == 0) {
      if (iarg+2 > narg) error->all(FLERR,"Illegal fix rigid/shell command");
      if (strcmp(arg[iarg+1],"xyz") == 0) pcouple = XYZ;
      else if (strcmp(arg[iarg+1],"xy") == 0) pcouple = XY;
      else if (strcmp(arg[iarg+1],"yz") == 0) pcouple = YZ;
      else if (strcmp(arg[iarg+1],"xz") == 0) pcouple = XZ;
      else if (strcmp(arg[iarg+1],"none") == 0) pcouple = NONE;
      else error->all(FLERR,"Illegal fix rigid/shell command");
      iarg += 2;

    } else if (strcmp(arg[iarg],"dilate") == 0) {
      if (iarg+2 > narg)
        error->all(FLERR,"Illegal fix rigid/shell nvt/npt/nph command");
      if (strcmp(arg[iarg+1],"all") == 0) allremap = 1;
      else {
        allremap = 0;
        delete[] id_dilate;
        id_dilate = utils::strdup(arg[iarg+1]);
        int idilate = group->find(id_dilate);
        if (idilate == -1)
          error->all(FLERR,"Fix rigid/shell nvt/npt/nph dilate group ID "
                     "does not exist");
      }
      iarg += 2;

    } else if (strcmp(arg[iarg],"tparam") == 0) {
      if (iarg+4 > narg) error->all(FLERR,"Illegal fix rigid/shell command");
      if (!utils::strmatch(style,"^rigid/n.t/shell"))
        error->all(FLERR,"Illegal fix rigid/shell command");
      t_chain = utils::inumeric(FLERR,arg[iarg+1],false,lmp);
      t_iter = utils::inumeric(FLERR,arg[iarg+2],false,lmp);
      t_order = utils::inumeric(FLERR,arg[iarg+3],false,lmp);
      iarg += 4;

    } else if (strcmp(arg[iarg],"pchain") == 0) {
      if (iarg+2 > narg) error->all(FLERR,"Illegal fix rigid/shell command");
      if (!utils::strmatch(style,"^rigid/np./shell"))
        error->all(FLERR,"Illegal fix rigid/shell command");
      p_chain = utils::inumeric(FLERR,arg[iarg+1],false,lmp);
      iarg += 2;

    } else if (strcmp(arg[iarg],"gravity") == 0) {
      if (iarg+2 > narg) error->all(FLERR,"Illegal fix rigid/shell command");
      delete[] id_gravity;
      id_gravity = utils::strdup(arg[iarg+1]);
      iarg += 2;

    } else error->all(FLERR,"Illegal fix rigid/shell command");
  }

  // error check and further setup for Molecule template

  if (onemols) {
    for (i = 0; i < nmol; i++) {
      if (onemols[i]->xflag == 0)
        error->all(FLERR,"Fix rigid/shell molecule must have coordinates");
      if (onemols[i]->typeflag == 0)
        error->all(FLERR,"Fix rigid/shell molecule must have atom types");

      // fix rigid/shell uses center, masstotal, COM, inertia of molecule

      onemols[i]->compute_center();
      onemols[i]->compute_mass();
      onemols[i]->compute_com();
      onemols[i]->compute_inertia();
    }
  }

  // set pstat_flag

  pstat_flag = 0;
  for (i = 0; i < 3; i++)
    if (p_flag[i]) pstat_flag = 1;

  if (pcouple == XYZ || (domain->dimension == 2 && pcouple == XY)) pstyle = ISO;
  else pstyle = ANISO;

  // create rigid bodies based on molecule or custom ID
  // sets bodytag for owned atoms
  // body attributes are computed later by setup_bodies()

  double time1 = platform::walltime();

  create_bodies(bodyID);
  if (customflag) delete[] bodyID;

  if (comm->me == 0)
    utils::logmesg(lmp,"  create bodies CPU = {:.3f} seconds\n", platform::walltime()-time1);

  // set nlocal_body and allocate bodies I own

  tagint *tag = atom->tag;

  nlocal_body = nghost_body = 0;
  for (i = 0; i < nlocal; i++)
    if (bodytag[i] == tag[i]) nlocal_body++;

  nmax_body = 0;
  while (nmax_body < nlocal_body) nmax_body += DELTA_BODY;
  body = (Body *) memory->smalloc(nmax_body*sizeof(Body), "rigid/shell:body");

  // set bodyown for owned atoms

  nlocal_body = 0;
  for (i = 0; i < nlocal; i++)
    if (bodytag[i] == tag[i]) {
      body[nlocal_body].ilocal = i;
      bodyown[i] = nlocal_body++;
    } else bodyown[i] = -1;


  // bodysize = sizeof(Body) in doubles

  bodysize = sizeof(Body)/sizeof(double);
  if (bodysize*sizeof(double) != sizeof(Body)) bodysize++;

  // set max comm sizes needed by this fix

  comm_forward = 1 + bodysize;
  comm_reverse = 6;

  // atom style pointers to particles that store extra info

  avec_ellipsoid = dynamic_cast<AtomVecEllipsoid *>(atom->style_match("ellipsoid"));
  avec_line = dynamic_cast<AtomVecLine *>(atom->style_match("line"));
  avec_tri = dynamic_cast<AtomVecTri *>(atom->style_match("tri"));

  // compute per body forces and torques inside final_integrate() by default

  earlyflag = 0;

  // print statistics

  int one = 0;
  bigint atomone = 0;
  for (i = 0; i < nlocal; i++) {
    if (bodyown[i] >= 0) one++;
    if (bodytag[i] > 0) atomone++;
  }
  MPI_Allreduce(&one,&nbody,1,MPI_INT,MPI_SUM,world);
  bigint atomall;
  MPI_Allreduce(&atomone,&atomall,1,MPI_LMP_BIGINT,MPI_SUM,world);

  if (me == 0) {
    utils::logmesg(lmp,"  {} rigid bodies with {} atoms\n"
                   "  {:.8} = max distance from body owner to body atom\n",
                   nbody,atomall,maxextent);
  }

  // initialize Marsaglia RNG with processor-unique seed

  maxlang = 0;
  langextra = nullptr;
  random = nullptr;
  if (langflag) random = new RanMars(lmp,seed + comm->me);

  // mass vector for granular pair styles

  mass_body = nullptr;
  nmax_mass = 0;

  // wait to setup bodies until comm stencils are defined

  setupflag = 0;

  // set hardness components once and for all if CONSTANT

  varflag = CONSTANT;
  if (hstyle != CONSTANT || mustyle != CONSTANT) varflag = EQUAL;
}

/* ---------------------------------------------------------------------- */

FixRigidShell::~FixRigidShell()
{
  // unregister callbacks to this fix from Atom class

  atom->delete_callback(id,Atom::GROW);

  // delete locally stored arrays

  memory->sfree(body);

  memory->destroy(bodyown);
  memory->destroy(bodytag);
  memory->destroy(atom2body);
  memory->destroy(angle2body);
  memory->destroy(vertexdata);
  memory->destroy(xcmimage);
  memory->destroy(displace);
  memory->destroy(eflags);
  memory->destroy(orient);
  memory->destroy(dorient);

  delete random;
  delete[] inpfile;
  delete[] id_dilate;
  delete[] id_gravity;

  memory->destroy(langextra);
  memory->destroy(mass_body);
}

/* ---------------------------------------------------------------------- */

int FixRigidShell::setmask()
{
  int mask = 0;
  mask |= INITIAL_INTEGRATE;
  mask |= FINAL_INTEGRATE;
  mask |= POST_FORCE;
  // if (langflag) mask |= POST_FORCE;
  mask |= PRE_NEIGHBOR;
  mask |= INITIAL_INTEGRATE_RESPA;
  mask |= FINAL_INTEGRATE_RESPA;
  return mask;
}

/* ---------------------------------------------------------------------- */

void FixRigidShell::init()
{
  triclinic = domain->triclinic;

  // need a full perpetual neighbor list
  neighbor->add_request(this,NeighConst::REQ_FULL);
  
  // warn if more than one rigid fix
  // if earlyflag, warn if any post-force fixes come after a rigid fix

  int count = 0;
  for (auto &ifix : modify->get_fix_list())
    if (ifix->rigid_flag) count++;
  if (count > 1 && me == 0) error->warning(FLERR,"More than one fix rigid");

  if (earlyflag) {
    bool rflag = false;
    for (auto &ifix : modify->get_fix_list()) {
      if (ifix->rigid_flag) rflag = true;
      if ((comm->me == 0) && rflag && (ifix->setmask() & POST_FORCE) && !ifix->rigid_flag)
        error->warning(FLERR,"Fix {} with ID {} alters forces after fix rigid/shell",
                       ifix->style, ifix->id);
    }
  }
  // check variables

    if (hstr) {
      hvar = input->variable->find(hstr);
      if (hvar < 0)
        error->all(FLERR,"Variable name for fix rigid/shell does not exist");
      if (!input->variable->equalstyle(hvar))
        error->all(FLERR,"Variable for fix rigid/shell is invalid style");
    }
    if (mustr) {
      muvar = input->variable->find(mustr);
      if (muvar < 0)
        error->all(FLERR,"Variable name for fix rigid/shell does not exist");
      if (!input->variable->equalstyle(muvar))
        error->all(FLERR,"Variable for fix rigid/shell is invalid style");
    }
    
  // warn if body properties are read from inpfile or a mol template file
  //   and the gravity keyword is not set and a gravity fix exists
  // this could mean body particles are overlapped
  //   and gravity is not applied correctly

  if ((inpfile || onemols) && !id_gravity) {
    if (modify->get_fix_by_style("^gravity").size() > 0)
      if (comm->me == 0)
        error->warning(FLERR,"Gravity may not be correctly applied to rigid "
                       "bodies if they consist of overlapped particles");
  }

  // error if a fix changing the box comes before rigid fix

  bool boxflag = false;
  for (auto &ifix : modify->get_fix_list()) {
    if (boxflag && utils::strmatch(ifix->style,"^rigid"))
        error->all(FLERR,"Rigid fixes must come before any box changing fix");
    if (ifix->box_change) boxflag = true;
  }

  // add gravity forces based on gravity vector from fix

  if (id_gravity) {
    auto ifix = modify->get_fix_by_id(id_gravity);
    if (!ifix) error->all(FLERR,"Fix rigid/shell cannot find fix gravity ID {}", id_gravity);
    if (!utils::strmatch(ifix->style,"^gravity"))
      error->all(FLERR,"Fix rigid/shell gravity fix ID {} is not a gravity fix style", id_gravity);
    int tmp;
    gvec = (double *) ifix->extract("gvec", tmp);
  }

  // timestep info

  dtv = update->dt;
  dtf = 0.5 * update->dt * force->ftm2v;
  dtq = 0.5 * update->dt;

  if (utils::strmatch(update->integrate_style,"^respa"))
    step_respa = (dynamic_cast<Respa *>(update->integrate))->step;
}

/* ---------------------------------------------------------------------- */

void FixRigidShell::init_list(int /*id*/, NeighList *ptr)
{
  list = ptr;
}

  /* ---------------------------------------------------------------------- */

/* ----------------------------------------------------------------------
   setup static/dynamic properties of rigid bodies, using current atom info.
   if reinitflag is not set, do the initialization only once, b/c properties
   may not be re-computable especially if overlapping particles or bodies
   are inserted from mol template.
     do not do dynamic init if read body properties from inpfile. this
   is b/c the inpfile defines the static and dynamic properties and may not
   be computable if contain overlapping particles setup_bodies_static()
   reads inpfile itself.
     cannot do this until now, b/c requires comm->setup() to have setup stencil
   invoke pre_neighbor() to ensure body xcmimage flags are reset
     needed if Verlet::setup::pbc() has remapped/migrated atoms for 2nd run
     setup_bodies_static() invokes pre_neighbor itself
------------------------------------------------------------------------- */

void FixRigidShell::setup_pre_neighbor()
{
  if (reinitflag || !setupflag){
    std::cout << "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ THIS Calling build_topology() ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~" << std::endl;
    neighbor->build_topology();
    std::cout << "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ : Calling setup_bodies_static() ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~" << std::endl;
    setup_bodies_static();
  }
  else pre_neighbor();

  if ((reinitflag || !setupflag) && !inpfile)
    setup_bodies_dynamic();

  setupflag = 1;
}

/* ----------------------------------------------------------------------
   compute initial fcm and torque on bodies, also initial virial
   reset all particle velocities to be consistent with vcm and omega
------------------------------------------------------------------------- */

void FixRigidShell::setup(int vflag)
{
  int i,n,ibody;

  // error if maxextent > comm->cutghost
  // NOTE: could just warn if an override flag set
  // NOTE: this could fail for comm multi mode if user sets a wrong cutoff
  //       for atom types in rigid bodies - need a more careful test
  // must check here, not in init, b/c neigh/comm values set after fix init

  double cutghost = MAX(neighbor->cutneighmax,comm->cutghostuser);
  if (maxextent > cutghost)
    error->all(FLERR,"Rigid body extent > ghost cutoff - use comm_modify cutoff");

  //check(1);

  // sum fcm, torque across all rigid bodies
  // fcm = force on COM
  // torque = torque around COM

  double **x = atom->x;
  double **f = atom->f;
  int nlocal = atom->nlocal;

  double *xcm,*fcm,*tcm;
  double dx,dy,dz;
  double unwrap[3];

  for (ibody = 0; ibody < nlocal_body+nghost_body; ibody++) {
    fcm = body[ibody].fcm;
    fcm[0] = fcm[1] = fcm[2] = 0.0;
    tcm = body[ibody].torque;
    tcm[0] = tcm[1] = tcm[2] = 0.0;
  }

  for (i = 0; i < nlocal; i++) {
    if (atom2body[i] < 0) continue;
    Body *b = &body[atom2body[i]];

    fcm = b->fcm;
    fcm[0] += f[i][0];
    fcm[1] += f[i][1];
    fcm[2] += f[i][2];

    domain->unmap(x[i],xcmimage[i],unwrap);
    xcm = b->xcm;
    dx = unwrap[0] - xcm[0];
    dy = unwrap[1] - xcm[1];
    dz = unwrap[2] - xcm[2];

    tcm = b->torque;
    tcm[0] += dy * f[i][2] - dz * f[i][1];
    tcm[1] += dz * f[i][0] - dx * f[i][2];
    tcm[2] += dx * f[i][1] - dy * f[i][0];
  }

  // extended particles add their rotation/torque to angmom/torque of body

  if (extended) {
    double **torque = atom->torque;

    for (i = 0; i < nlocal; i++) {
      if (atom2body[i] < 0) continue;
      Body *b = &body[atom2body[i]];
      if (eflags[i] & TORQUE) {
        tcm = b->torque;
        tcm[0] += torque[i][0];
        tcm[1] += torque[i][1];
        tcm[2] += torque[i][2];
      }
    }
  }

  // reverse communicate fcm, torque of all bodies

  commflag = FORCE_TORQUE;
  comm->reverse_comm(this,6);

  // virial setup before call to set_v

  v_init(vflag);

  // compute and forward communicate vcm and omega of all bodies

  for (ibody = 0; ibody < nlocal_body; ibody++) {
    Body *b = &body[ibody];
    MathExtra::angmom_to_omega(b->angmom,b->ex_space,b->ey_space,
                               b->ez_space,b->inertia,b->omega);
  }

  commflag = FINAL;
  comm->forward_comm(this,10);

  // set velocity/rotation of atoms in rigid bodues

  set_v();

  // guesstimate virial as 2x the set_v contribution

  if (vflag_global)
    for (n = 0; n < 6; n++) virial[n] *= 2.0;
  if (vflag_atom) {
    for (i = 0; i < nlocal; i++)
      for (n = 0; n < 6; n++)
        vatom[i][n] *= 2.0;
  }
}

/* ---------------------------------------------------------------------- */

void FixRigidShell::initial_integrate(int vflag)
{
  areas_and_normals();
  double dtfm;

  //check(2);

  for (int ibody = 0; ibody < nlocal_body; ibody++) {
    Body *b = &body[ibody];

    // update vcm by 1/2 step

    dtfm = dtf / b->mass;
    b->vcm[0] += dtfm * b->fcm[0];
    b->vcm[1] += dtfm * b->fcm[1];
    b->vcm[2] += dtfm * b->fcm[2];

    // update xcm by full step

    b->xcm[0] += dtv * b->vcm[0];
    b->xcm[1] += dtv * b->vcm[1];
    b->xcm[2] += dtv * b->vcm[2];

    // update angular momentum by 1/2 step

    b->angmom[0] += dtf * b->torque[0];
    b->angmom[1] += dtf * b->torque[1];
    b->angmom[2] += dtf * b->torque[2];

    // compute omega at 1/2 step from angmom at 1/2 step and current q
    // update quaternion a full step via Richardson iteration
    // returns new normalized quaternion, also updated omega at 1/2 step
    // update ex,ey,ez to reflect new quaternion

    MathExtra::angmom_to_omega(b->angmom,b->ex_space,b->ey_space,
                               b->ez_space,b->inertia,b->omega);
    MathExtra::richardson(b->quat,b->angmom,b->omega,b->inertia,dtq);
    MathExtra::q_to_exyz(b->quat,b->ex_space,b->ey_space,b->ez_space);
  }

  // virial setup before call to set_xv

  v_init(vflag);

  // forward communicate updated info of all bodies

  commflag = INITIAL;
  comm->forward_comm(this,29);


  // set coords/orient and velocity/rotation of atoms in rigid bodies and displace atoms within the bodies
  set_xv();
}

/* ----------------------------------------------------------------------
   apply Langevin thermostat to all 6 DOF of rigid bodies I own
   unlike fix langevin, this stores extra force in extra arrays,
     which are added in when a new fcm/torque are calculated
------------------------------------------------------------------------- */

void FixRigidShell::apply_langevin_thermostat()
{
  double gamma1,gamma2;
  double wbody[3],tbody[3];

  // grow langextra if needed

  if (nlocal_body > maxlang) {
    memory->destroy(langextra);
    maxlang = nlocal_body + nghost_body;
    memory->create(langextra,maxlang,6,"rigid/shell:langextra");
  }

  double delta = update->ntimestep - update->beginstep;
  delta /= update->endstep - update->beginstep;
  double t_target = t_start + delta * (t_stop-t_start);
  double tsqrt = sqrt(t_target);

  double boltz = force->boltz;
  double dt = update->dt;
  double mvv2e = force->mvv2e;
  double ftm2v = force->ftm2v;

  double *vcm,*omega,*inertia,*ex_space,*ey_space,*ez_space;

  for (int ibody = 0; ibody < nlocal_body; ibody++) {
    vcm = body[ibody].vcm;
    omega = body[ibody].omega;
    inertia = body[ibody].inertia;
    ex_space = body[ibody].ex_space;
    ey_space = body[ibody].ey_space;
    ez_space = body[ibody].ez_space;

    gamma1 = -body[ibody].mass / t_period / ftm2v;
    gamma2 = sqrt(body[ibody].mass) * tsqrt *
      sqrt(24.0*boltz/t_period/dt/mvv2e) / ftm2v;
    langextra[ibody][0] = gamma1*vcm[0] + gamma2*(random->uniform()-0.5);
    langextra[ibody][1] = gamma1*vcm[1] + gamma2*(random->uniform()-0.5);
    langextra[ibody][2] = gamma1*vcm[2] + gamma2*(random->uniform()-0.5);

    gamma1 = -1.0 / t_period / ftm2v;
    gamma2 = tsqrt * sqrt(24.0*boltz/t_period/dt/mvv2e) / ftm2v;

    // convert omega from space frame to body frame

    MathExtra::transpose_matvec(ex_space,ey_space,ez_space,omega,wbody);

    // compute langevin torques in the body frame

    tbody[0] = inertia[0]*gamma1*wbody[0] +
      sqrt(inertia[0])*gamma2*(random->uniform()-0.5);
    tbody[1] = inertia[1]*gamma1*wbody[1] +
      sqrt(inertia[1])*gamma2*(random->uniform()-0.5);
    tbody[2] = inertia[2]*gamma1*wbody[2] +
      sqrt(inertia[2])*gamma2*(random->uniform()-0.5);

    // convert langevin torques from body frame back to space frame

    MathExtra::matvec(ex_space,ey_space,ez_space,tbody,&langextra[ibody][3]);

    // enforce 2d motion

    if (domain->dimension == 2)
      langextra[ibody][2] = langextra[ibody][3] = langextra[ibody][4] = 0.0;
  }
}

/* ----------------------------------------------------------------------
   called from FixEnforce post_force() for 2d problems
   zero all body values that should be zero for 2d model
------------------------------------------------------------------------- */

void FixRigidShell::enforce2d()
{
  Body *b;

  for (int ibody = 0; ibody < nlocal_body; ibody++) {
    b = &body[ibody];
    b->xcm[2] = 0.0;
    b->vcm[2] = 0.0;
    b->fcm[2] = 0.0;
    b->xgc[2] = 0.0;
    b->torque[0] = 0.0;
    b->torque[1] = 0.0;
    b->angmom[0] = 0.0;
    b->angmom[1] = 0.0;
    b->omega[0] = 0.0;
    b->omega[1] = 0.0;
    if (langflag && langextra) {
      langextra[ibody][2] = 0.0;
      langextra[ibody][3] = 0.0;
      langextra[ibody][4] = 0.0;
    }
  }
}

/* ---------------------------------------------------------------------- */

void FixRigidShell::post_force(int /*vflag*/)
{
  if (langflag) apply_langevin_thermostat();
  if (earlyflag) compute_forces_and_torques();

 // update hardness due to variables

  if (varflag != CONSTANT) {
    modify->clearstep_compute();
    if (hstyle == EQUAL) pf = input->variable->compute_equal(hvar);
    if (mustyle == EQUAL) mu = input->variable->compute_equal(muvar);
    modify->addstep_compute(update->ntimestep + 1);
  }


  // Call abrasion functions
  
  double **f = atom->f;
  double v_rel[3];
	double x_rel[3];
	double speed_of_approach;
  bool force_acting;

  int local_index;
  int no_of_neighs;
  int neigh_index;
  //  NeighList *l = neighbor->lists[list_index];
  // std::cout << "inum: " << list->inum << std::endl;
  // std::cout << "nlocal: " << atom->nlocal << std::endl;
  //  for (int i = 0; i < list->inum; i++) {
  for (int i = 0; i < atom->nlocal; i++) {
    force_acting = MathExtra::len3(f[i]);
    if (force_acting) {
      local_index = list->ilist[i];
      no_of_neighs = list->numneigh[i];
      // std::cout << "atom: " << i << ", local index: " << local_index << ", numneigh: "
		// << list->numneigh[i] << std::endl;
      for (int j = 0; j < list->numneigh[i]; j++) {
	neigh_index = list->firstneigh[i][j];
 

 
  if (atom2body[neigh_index] == atom2body[local_index]){
    continue;}

  // A displacement occurs only if the objects are approaching each other.
	if (gap_is_shrinking(local_index, neigh_index, v_rel, x_rel, &speed_of_approach)) {
    displacement_of_atom(i, (atom->radius)[neigh_index], speed_of_approach, v_rel, x_rel);
      	}
      }
    }
  }
  

}


/* ---------------------------------------------------------------------- */

void FixRigidShell::areas_and_normals() {

  int i1, i2, i3, n, type;
  double delx1, dely1, delz1, delx2, dely2, delz2;
  double eangle, f1[3], f3[3];
  double rsq1, rsq2, r1, r2, c, a, a11, a12, a22;
  double axbi, axbj, axbk, area;
  double centroid[3], se1[3], se2[3], se3[3];
  double st[3], dots[3], abs[3];
  double sub_area;
  double norm1, norm2, norm3, length;
  double n1, n2, n3;

  //  std::cout << "Entering areas_and_normals()" << std::endl;

  double **x = atom->x;
  double **f = atom->f;
  int **anglelist = neighbor->anglelist;
  int nanglelist = neighbor->nanglelist;
  int nlocal = atom->nlocal;
  int newton_bond = force->newton_bond;

  for (int i = 0; i < nlocal; i++) {
    vertexdata[i][0] = 0.0; // x normal
    vertexdata[i][1] = 0.0; // y normal
    vertexdata[i][2] = 0.0; // z normal
    vertexdata[i][3] = 0.0; // associated area
    vertexdata[i][4] = 0.0; // x displacement velocity
    vertexdata[i][5] = 0.0; // y displacement velocity
    vertexdata[i][6] = 0.0; // z displacement velocity
  }

  norm1 = 0.0;
  norm2 = 0.0;
  norm3 = 0.0;
  sub_area = 0.0;

  //  Storing each atom in each angle - Not currently specified to a single body (potential optimisation)
  for (n = 0; n < nanglelist; n++) {
    i1 = anglelist[n][0];
    i2 = anglelist[n][1];
    i3 = anglelist[n][2];
    type = anglelist[n][3];

    // 1st bond

    delx1 = displace[i1][0] - displace[i2][0];
    dely1 = displace[i1][1] - displace[i2][1];
    delz1 = displace[i1][2] - displace[i2][2];

    rsq1 = delx1 * delx1 + dely1 * dely1 + delz1 * delz1;
    r1 = sqrt(rsq1);

    // 2nd bond

    delx2 = displace[i3][0] - displace[i2][0];
    dely2 = displace[i3][1] - displace[i2][1];
    delz2 = displace[i3][2] - displace[i2][2];

    rsq2 = delx2 * delx2 + dely2 * dely2 + delz2 * delz2;
    r2 = sqrt(rsq2);

    // cross product
    // a x b = (a2.b3 - a3.b2)i + (a3.b1 - a1.b3)j + (a1.b2 - a2.b1)k
    // b is the first bond whilst a is the second bond.

    axbi = dely2*delz1 - delz2*dely1;
    axbj = delz2*delx1 - delx2*delz1;
    axbk = delx2*dely1 - dely2*delx1;

    // area of facet

    area = sqrt(axbi*axbi + axbj*axbj + axbk*axbk); // actually 2*area

    n1 = axbi/area;
    n2 = axbj/area;
    n3 = axbk/area;

    //    std::cout << "Normal to facet: " << n1 << " " << n2 << " " << n3 << std::endl;
    //    std::cout << "Area of facet: " << area/2 << std::endl;

    // Centroid of the vertices making the current triangle
    centroid[0] = (displace[i1][0] + displace[i2][0] + displace[i3][0])/3.0;
    centroid[1] = (displace[i1][1] + displace[i2][1] + displace[i3][1])/3.0;
    centroid[2] = (displace[i1][2] + displace[i2][2] + displace[i3][2])/3.0;
    //    std::cout << "Centroid: " << centroid[0] << " " << centroid[1]
    //	      << " " << centroid[2] << std::endl;


    // // CHECK IF THE DOT PRODUCT WITH THE COM) IS < 0 - CHECKS IF NORMALS ARE INVERTED

    // since we are using body coordinates, [0,0,0] is the COM?
    
    if  ((((centroid[0] - 0) * n1) + ((centroid[1] - 0) * n2) + ((centroid[2] - 0) * n3)) < 0) {
      // Flip normal if it is pointing the wrong way
      n1 = -n1;
      n2 = -n2;
      n3 = -n3;
    }
    // Sub-edge 1
    se1[0] = displace[i1][0] - centroid[0];
    se1[1] = displace[i1][1] - centroid[1];
    se1[2] = displace[i1][2] - centroid[2];

    // Sub-edge 2
    se2[0] = displace[i2][0] - centroid[0];
    se2[1] = displace[i2][1] - centroid[1];
    se2[2] = displace[i2][2] - centroid[2];

    // Sub-edge 3
    se3[0] = displace[i3][0] - centroid[0];
    se3[1] = displace[i3][1] - centroid[1];
    se3[2] = displace[i3][2] - centroid[2];

    // dots between sub-edges 1-2 2-3 3-1
    dots[0] = MathExtra::dot3(se1, se2);
    dots[1] = MathExtra::dot3(se2, se3);
    dots[2] = MathExtra::dot3(se3, se1);

    // absolute length of sub-edges
    abs[0] = MathExtra::len3(se1);
    abs[1] = MathExtra::len3(se2);
    abs[2] = MathExtra::len3(se3);

    // sin of the angle between sub-edges (from centroid to vertices)
    // sin(theta) = sqrt(1 - cos(theta)^2), cos(theta) = dots / abs
    st[0] = std::sqrt(1.0 - std::pow(dots[0]/(abs[0]*abs[1]),2));
    st[1] = std::sqrt(1.0 - std::pow(dots[1]/(abs[1]*abs[2]),2));
    st[2] = std::sqrt(1.0 - std::pow(dots[2]/(abs[2]*abs[0]),2));

    // Half of each sub-triangle associated with each vertex
    // A = 0.5 * se1 * se2 * st
    //    std::cout << "i1: " << i1 << std::endl;
    //    std::cout << "rmass: " << atom->rmass[i1] << std::endl;
    //    std::cout << "radius: " << (atom->radius)[i1] << std::endl;

    sub_area = 0.25 * abs[0] * abs[1] * st[0];
    vertexdata[i1][3] += sub_area;
    vertexdata[i1][0] += n1*sub_area;
    vertexdata[i1][1] += n2*sub_area;
    vertexdata[i1][2] += n3*sub_area;
    vertexdata[i2][3] += sub_area;
    vertexdata[i2][0] += n1*sub_area;
    vertexdata[i2][1] += n2*sub_area;
    vertexdata[i2][2] += n3*sub_area;

    sub_area = 0.25 * abs[1] * abs[2] * st[1];
    vertexdata[i2][3] += sub_area;
    vertexdata[i2][0] += n1*sub_area;
    vertexdata[i2][1] += n2*sub_area;
    vertexdata[i2][2] += n3*sub_area;
    vertexdata[i3][3] += sub_area;
    vertexdata[i3][0] += n1*sub_area;
    vertexdata[i3][1] += n2*sub_area;
    vertexdata[i3][2] += n3*sub_area;

    sub_area = 0.25 * abs[2] * abs[0] * st[2];
    vertexdata[i3][3] += sub_area;
    vertexdata[i3][0] += n1*sub_area;
    vertexdata[i3][1] += n2*sub_area;
    vertexdata[i3][2] += n3*sub_area;
    vertexdata[i1][3] += sub_area;
    vertexdata[i1][0] += n1*sub_area;
    vertexdata[i1][1] += n2*sub_area;
    vertexdata[i1][2] += n3*sub_area;

  }

  for (int i = 0; i < nlocal; i++) {
    if (vertexdata[i][3] > 0) {
      norm1 = vertexdata[i][0];
      norm2 = vertexdata[i][1];
      norm3 = vertexdata[i][2];
      length = sqrt(norm1*norm1 + norm2*norm2 + norm3*norm3);
      vertexdata[i][0] = norm1/length;
      vertexdata[i][1] = norm2/length;
      vertexdata[i][2] = norm3/length;
    } else {
      vertexdata[i][0] = 0.0;
      vertexdata[i][1] = 0.0;
      vertexdata[i][2] = 0.0;
    }
    // std::cout << "Constructed area associated with atom " << i << " is " << vertexdata[i][3]
	  //     << ". Normal to constructed area is: " << vertexdata[i][0]
	  //     << " " << vertexdata[i][1] << " " << vertexdata[i][2] << std::endl;
  }
}

/* ---------------------------------------------------------------------- */

bool FixRigidShell::gap_is_shrinking(int i, int neigh_index,
				   double v_rel[3], double x_rel[3],
				   double *vnorm_p) {

  double **v = atom->v;
  double **x = atom->x;

  // std::cout << "neighbour vx, vy, vz: " << v[neigh_index][0] << ", " << v[neigh_index][1]
	//     << ", " << v[neigh_index][2] << std::endl;

  // std::cout << "neighbour x: " << x[neigh_index][0] << ", y: " << x[neigh_index][1]
	//     << ", z: " << x[neigh_index][2] << std::endl;

  v_rel[0] = v[neigh_index][0] - v[i][0];
  v_rel[1] = v[neigh_index][1] - v[i][1];
  v_rel[2] = v[neigh_index][2] - v[i][2];

  double normal[3];
  normal[0] = vertexdata[i][0];
  normal[1] = vertexdata[i][1];
  normal[2] = vertexdata[i][2];

  *vnorm_p = MathExtra::dot3(v_rel, normal);
  // std::cout << "vnorm = " << *vnorm_p
	//     << ". vnorm < 0 indicates that the gap is shrinking." << std::endl;

  return *vnorm_p < 0;
}

/* ---------------------------------------------------------------------- */

void FixRigidShell::displacement_of_atom(int i, double r,
				       double vnorm,
				       double v_rel[3], double x_rel[3]) {


  // double pf = 10000000.0;                    // hardness
  // double mu = 0.25;                          // a constant used to calculate the shear hardness
  double **f = atom->f;
  double normal[3];
  normal[0] = vertexdata[i][0];
  normal[1] = vertexdata[i][1];
  normal[2] = vertexdata[i][2];
  double area = vertexdata[i][3];

  double normal_component_of_force =  MathExtra::dot3(normal, f[i]);
  double tangential_force[3];
  tangential_force[0] = f[i][0] - normal_component_of_force*normal[0];
  tangential_force[1] = f[i][1] - normal_component_of_force*normal[1];
  tangential_force[2] = f[i][2] - normal_component_of_force*normal[2];
  double magnitude_of_tangential_force = MathExtra::len3(tangential_force);
  double dt = update->dt;

  double separation_of_centres = MathExtra::dot3(x_rel, normal);
  // std::cout << "separation of centres = " << separation_of_centres << std::endl;

  // Effective radius.
  double r_eff = r + (atom->radius)[i];
  // std::cout << "r =  " << r << ", r_eff =" << r_eff << std::endl;

  // Overlap of spheres.
  double h = r_eff - separation_of_centres;
  // std::cout << "overlap = " << h << std::endl;
  
  // Tangential velocity.
  double normal_component_of_vel =  MathExtra::dot3(normal, v_rel);
  double v_tan[3];
  v_tan[0] = v_rel[0] -  normal_component_of_vel*normal[0];
  v_tan[1] = v_rel[1] -  normal_component_of_vel*normal[1];
  v_tan[2] = v_rel[2] -  normal_component_of_vel*normal[2];

  // Tangential displacement.
  double deltas = MathExtra::len3(v_tan)*dt;

  double deltah = 0.0;
  // std::cout << "normal_component_of_force = " << normal_component_of_force << std::endl;
  // std::cout << "area*pf = " << area*pf << std::endl;

  // Check if theres indentation
  bool iV = abs(normal_component_of_force) > area*pf;

  if (iV) {
    deltah = vnorm*dt;
    // std::cout << "deltah = " << deltah << std::endl; 
  }

  // This is the normal displacement corresponding to deltas.
  double dsh = 0.0;

  // Check if theres scratching
  bool iL = magnitude_of_tangential_force > area*mu*pf;

  bool overlapping = h > 0;
  if (iL && overlapping) {
    dsh = h-r_eff+sqrt((r_eff-h)*(r_eff-h) + 2*sqrt(abs(2*r_eff*h-h*h))*deltas);
    // std::cout << "dsh = " << dsh << ", deltas = " << deltah << std::endl;
  }

  // Total normal displacement.
  double total_normal_dispacement  = deltah - dsh;
  // std::cout << "total_normal_dispacement = " << total_normal_dispacement << std::endl;
  double displ_speed = total_normal_dispacement/dt;
  vertexdata[i][4] += displ_speed*normal[0]; // x velocity for displacement
  vertexdata[i][5] += displ_speed*normal[1]; // y velocity for displacement
  vertexdata[i][6] += displ_speed*normal[2]; // z velocity for displacement
  // std::cout << "vx: " << vertexdata[i][4] << std::endl;
  // std::cout << "vy: " << vertexdata[i][5] << std::endl;
  // std::cout << "vz: " << vertexdata[i][6] << std::endl;
}

/* ---------------------------------------------------------------------- */

void FixRigidShell::compute_forces_and_torques()
{
  int i,ibody;

  //check(3);

  // sum over atoms to get force and torque on rigid body

  double **x = atom->x;
  double **f = atom->f;
  int nlocal = atom->nlocal;

  double dx,dy,dz;
  double unwrap[3];
  double *xcm,*fcm,*tcm;

  for (ibody = 0; ibody < nlocal_body+nghost_body; ibody++) {
    fcm = body[ibody].fcm;
    fcm[0] = fcm[1] = fcm[2] = 0.0;
    tcm = body[ibody].torque;
    tcm[0] = tcm[1] = tcm[2] = 0.0;
  }

  for (i = 0; i < nlocal; i++) {
    if (atom2body[i] < 0) continue;
    Body *b = &body[atom2body[i]];

    fcm = b->fcm;
    fcm[0] += f[i][0];
    fcm[1] += f[i][1];
    fcm[2] += f[i][2];

    domain->unmap(x[i],xcmimage[i],unwrap);
    xcm = b->xcm;
    dx = unwrap[0] - xcm[0];
    dy = unwrap[1] - xcm[1];
    dz = unwrap[2] - xcm[2];

    tcm = b->torque;
    tcm[0] += dy*f[i][2] - dz*f[i][1];
    tcm[1] += dz*f[i][0] - dx*f[i][2];
    tcm[2] += dx*f[i][1] - dy*f[i][0];
  }

  // extended particles add their torque to torque of body

  if (extended) {
    double **torque = atom->torque;

    for (i = 0; i < nlocal; i++) {
      if (atom2body[i] < 0) continue;

      if (eflags[i] & TORQUE) {
        tcm = body[atom2body[i]].torque;
        tcm[0] += torque[i][0];
        tcm[1] += torque[i][1];
        tcm[2] += torque[i][2];
      }
    }
  }

  // reverse communicate fcm, torque of all bodies

  commflag = FORCE_TORQUE;
  comm->reverse_comm(this,6);

  // include Langevin thermostat forces and torques

  if (langflag) {
    for (ibody = 0; ibody < nlocal_body; ibody++) {
      fcm = body[ibody].fcm;
      fcm[0] += langextra[ibody][0];
      fcm[1] += langextra[ibody][1];
      fcm[2] += langextra[ibody][2];
      tcm = body[ibody].torque;
      tcm[0] += langextra[ibody][3];
      tcm[1] += langextra[ibody][4];
      tcm[2] += langextra[ibody][5];
    }
  }

  // add gravity force to COM of each body

  if (id_gravity) {
    double mass;
    for (ibody = 0; ibody < nlocal_body; ibody++) {
      mass = body[ibody].mass;
      fcm = body[ibody].fcm;
      fcm[0] += gvec[0]*mass;
      fcm[1] += gvec[1]*mass;
      fcm[2] += gvec[2]*mass;
    }
  }
}

/* ---------------------------------------------------------------------- */

void FixRigidShell::final_integrate()
{
  double dtfm;

  //check(3);

  if (!earlyflag) compute_forces_and_torques();

  // update vcm and angmom, recompute omega

  for (int ibody = 0; ibody < nlocal_body; ibody++) {
    Body *b = &body[ibody];

    // update vcm by 1/2 step

    dtfm = dtf / b->mass;
    b->vcm[0] += dtfm * b->fcm[0];
    b->vcm[1] += dtfm * b->fcm[1];
    b->vcm[2] += dtfm * b->fcm[2];

    // update angular momentum by 1/2 step

    b->angmom[0] += dtf * b->torque[0];
    b->angmom[1] += dtf * b->torque[1];
    b->angmom[2] += dtf * b->torque[2];

    MathExtra::angmom_to_omega(b->angmom,b->ex_space,b->ey_space,
                               b->ez_space,b->inertia,b->omega);
  }

  // forward communicate updated info of all bodies

  commflag = FINAL;
  comm->forward_comm(this,10);

  // set velocity/rotation of atoms in rigid bodies
  // virial is already setup from initial_integrate

  set_v();

    // Update displace array here to move atoms within the body
    // integrate position over the displacement velocity by a full step
  int nlocal = atom->nlocal;
  for (int i = 0; i < nlocal; i++) {
    if (atom2body[i] < 0) continue;
    displace[i][0] += dtv * vertexdata[i][4];
    displace[i][1] += dtv * vertexdata[i][5];
    displace[i][2] += dtv * vertexdata[i][6];
  }
}

/* ---------------------------------------------------------------------- */

void FixRigidShell::initial_integrate_respa(int vflag, int ilevel, int /*iloop*/)
{
  dtv = step_respa[ilevel];
  dtf = 0.5 * step_respa[ilevel] * force->ftm2v;
  dtq = 0.5 * step_respa[ilevel];

  if (ilevel == 0) initial_integrate(vflag);
  else final_integrate();
}

/* ---------------------------------------------------------------------- */

void FixRigidShell::final_integrate_respa(int ilevel, int /*iloop*/)
{
  dtf = 0.5 * step_respa[ilevel] * force->ftm2v;
  final_integrate();
}

/* ----------------------------------------------------------------------
   remap xcm of each rigid body back into periodic simulation box
   done during pre_neighbor so will be after call to pbc()
     and after fix_deform::pre_exchange() may have flipped box
   use domain->remap() in case xcm is far away from box
     due to first-time definition of rigid body in setup_bodies_static()
     or due to box flip
   also adjust imagebody = rigid body image flags, due to xcm remap
   then communicate bodies so other procs will know of changes to body xcm
   then adjust xcmimage flags of all atoms in bodies via image_shift()
     for two effects
     (1) change in true image flags due to pbc() call during exchange
     (2) change in imagebody due to xcm remap
   xcmimage flags are always -1,0,-1 so that body can be unwrapped
     around in-box xcm and stay close to simulation box
   if just inferred unwrapped from atom image flags,
     then a body could end up very far away
     when unwrapped by true image flags
   then set_xv() will compute huge displacements every step to reset coords of
     all the body atoms to be back inside the box, ditto for triclinic box flip
     note: so just want to avoid that numeric problem?
------------------------------------------------------------------------- */

void FixRigidShell::pre_neighbor()
{
  for (int ibody = 0; ibody < nlocal_body; ibody++) {
    Body *b = &body[ibody];
    domain->remap(b->xcm,b->image);
  }

  nghost_body = 0;
  commflag = FULL_BODY;
  comm->forward_comm(this);
  reset_atom2body();
  reset_angle2body();
  //check(4);

  image_shift();
}

/* ----------------------------------------------------------------------
   reset body xcmimage flags of atoms in bodies
   xcmimage flags are relative to xcm so that body can be unwrapped
   xcmimage = true image flag - imagebody flag
------------------------------------------------------------------------- */

void FixRigidShell::image_shift()
{
  imageint tdim,bdim,xdim[3];

  imageint *image = atom->image;
  int nlocal = atom->nlocal;

  for (int i = 0; i < nlocal; i++) {
    if (atom2body[i] < 0) continue;
    Body *b = &body[atom2body[i]];

    tdim = image[i] & IMGMASK;
    bdim = b->image & IMGMASK;
    xdim[0] = IMGMAX + tdim - bdim;
    tdim = (image[i] >> IMGBITS) & IMGMASK;
    bdim = (b->image >> IMGBITS) & IMGMASK;
    xdim[1] = IMGMAX + tdim - bdim;
    tdim = image[i] >> IMG2BITS;
    bdim = b->image >> IMG2BITS;
    xdim[2] = IMGMAX + tdim - bdim;

    xcmimage[i] = (xdim[2] << IMG2BITS) | (xdim[1] << IMGBITS) | xdim[0];
  }
}

/* ----------------------------------------------------------------------
   count # of DOF removed by rigid bodies for atoms in igroup
   return total count of DOF
------------------------------------------------------------------------- */

int FixRigidShell::dof(int tgroup)
{
  int i,j;

  // cannot count DOF correctly unless setup_bodies_static() has been called

  if (!setupflag) {
    if (comm->me == 0)
      error->warning(FLERR,"Cannot count rigid body degrees-of-freedom "
                     "before bodies are fully initialized");
    return 0;
  }

  int tgroupbit = group->bitmask[tgroup];

  // counts = 3 values per rigid body I own
  // 0 = # of point particles in rigid body and in temperature group
  // 1 = # of finite-size particles in rigid body and in temperature group
  // 2 = # of particles in rigid body, disregarding temperature group

  memory->create(counts,nlocal_body+nghost_body,3,"rigid/shell:counts");
  for (i = 0; i < nlocal_body+nghost_body; i++)
    counts[i][0] = counts[i][1] = counts[i][2] = 0;

  // tally counts from my owned atoms
  // 0 = # of point particles in rigid body and in temperature group
  // 1 = # of finite-size particles in rigid body and in temperature group
  // 2 = # of particles in rigid body, disregarding temperature group

  int *mask = atom->mask;
  int nlocal = atom->nlocal;

  for (i = 0; i < nlocal; i++) {
    if (atom2body[i] < 0) continue;
    j = atom2body[i];
    counts[j][2]++;
    if (mask[i] & tgroupbit) {
      if (extended && (eflags[i] & ~(POINT | DIPOLE))) counts[j][1]++;
      else counts[j][0]++;
    }
  }

  commflag = DOF;
  comm->reverse_comm(this,3);

  // nall = count0 = # of point particles in each rigid body
  // mall = count1 = # of finite-size particles in each rigid body
  // warn if nall+mall != nrigid for any body included in temperature group

  int flag = 0;
  for (int ibody = 0; ibody < nlocal_body; ibody++) {
    if (counts[ibody][0]+counts[ibody][1] > 0 &&
        counts[ibody][0]+counts[ibody][1] != counts[ibody][2]) flag = 1;
  }
  int flagall;
  MPI_Allreduce(&flag,&flagall,1,MPI_INT,MPI_MAX,world);
  if (flagall && me == 0)
    error->warning(FLERR,"Computing temperature of portions of rigid bodies");

  // remove appropriate DOFs for each rigid body wholly in temperature group
  // N = # of point particles in body
  // M = # of finite-size particles in body
  // 3d body has 3N + 6M dof to start with
  // 2d body has 2N + 3M dof to start with
  // 3d point-particle body with all non-zero I should have 6 dof, remove 3N-6
  // 3d point-particle body (linear) with a 0 I should have 5 dof, remove 3N-5
  // 2d point-particle body should have 3 dof, remove 2N-3
  // 3d body with any finite-size M should have 6 dof, remove (3N+6M) - 6
  // 2d body with any finite-size M should have 3 dof, remove (2N+3M) - 3

  double *inertia;

  int n = 0;
  nlinear = 0;
  if (domain->dimension == 3) {
    for (int ibody = 0; ibody < nlocal_body; ibody++) {
      if (counts[ibody][0]+counts[ibody][1] == counts[ibody][2]) {
        n += 3*counts[ibody][0] + 6*counts[ibody][1] - 6;
        inertia = body[ibody].inertia;
        if (inertia[0] == 0.0 || inertia[1] == 0.0 || inertia[2] == 0.0) {
          n++;
          nlinear++;
        }
      }
    }
  } else if (domain->dimension == 2) {
    for (int ibody = 0; ibody < nlocal_body; ibody++)
      if (counts[ibody][0]+counts[ibody][1] == counts[ibody][2])
        n += 2*counts[ibody][0] + 3*counts[ibody][1] - 3;
  }

  memory->destroy(counts);

  int nall;
  MPI_Allreduce(&n,&nall,1,MPI_INT,MPI_SUM,world);
  return nall;
}

/* ----------------------------------------------------------------------
   adjust xcm of each rigid body due to box deformation
   called by various fixes that change box size/shape
   flag = 0/1 means map from box to lamda coords or vice versa
------------------------------------------------------------------------- */

void FixRigidShell::deform(int flag)
{
  if (flag == 0)
    for (int ibody = 0; ibody < nlocal_body; ibody++)
      domain->x2lamda(body[ibody].xcm,body[ibody].xcm);
  else
    for (int ibody = 0; ibody < nlocal_body; ibody++)
      domain->lamda2x(body[ibody].xcm,body[ibody].xcm);
}

/* ----------------------------------------------------------------------
   set space-frame coords and velocity of each atom in each rigid body
   set orientation and rotation of extended particles
   x = Q displace + Xcm, mapped back to periodic box
   v = Vcm + (W cross (x - Xcm))
------------------------------------------------------------------------- */

void FixRigidShell::set_xv()
{
  

  int xbox,ybox,zbox;
  double x0,x1,x2,v0,v1,v2,fc0,fc1,fc2,massone;
  double ione[3],exone[3],eyone[3],ezone[3],vr[6],p[3][3];

  double xprd = domain->xprd;
  double yprd = domain->yprd;
  double zprd = domain->zprd;
  double xy = domain->xy;
  double xz = domain->xz;
  double yz = domain->yz;

  double **x = atom->x;
  double **v = atom->v;
  double **f = atom->f;
  double *rmass = atom->rmass;
  double *mass = atom->mass;
  int *type = atom->type;
  int nlocal = atom->nlocal;


  double v_temp = 0.1;

  // set x and v of each atom

  for (int i = 0; i < nlocal; i++) {
    if (atom2body[i] < 0) continue;
    Body *b = &body[atom2body[i]];

    xbox = (xcmimage[i] & IMGMASK) - IMGMAX;
    ybox = (xcmimage[i] >> IMGBITS & IMGMASK) - IMGMAX;
    zbox = (xcmimage[i] >> IMG2BITS) - IMGMAX;

    // save old positions and velocities for virial

    if (evflag) {
      if (triclinic == 0) {
        x0 = x[i][0] + xbox*xprd;
        x1 = x[i][1] + ybox*yprd;
        x2 = x[i][2] + zbox*zprd;
      } else {
        x0 = x[i][0] + xbox*xprd + ybox*xy + zbox*xz;
        x1 = x[i][1] + ybox*yprd + zbox*yz;
        x2 = x[i][2] + zbox*zprd;
      }
      v0 = v[i][0];
      v1 = v[i][1];
      v2 = v[i][2];
    }

    // x = displacement from center-of-mass, based on body orientation
    // v = vcm + omega around center-of-mass

    MathExtra::matvec(b->ex_space,b->ey_space,b->ez_space,displace[i],x[i]);

    v[i][0] = b->omega[1]*x[i][2] - b->omega[2]*x[i][1] + b->vcm[0];
    v[i][1] = b->omega[2]*x[i][0] - b->omega[0]*x[i][2] + b->vcm[1];
    v[i][2] = b->omega[0]*x[i][1] - b->omega[1]*x[i][0] + b->vcm[2];

    // add center of mass to displacement
    // map back into periodic box via xbox,ybox,zbox
    // for triclinic, add in box tilt factors as well

    if (triclinic == 0) {
      x[i][0] += b->xcm[0] - xbox*xprd;
      x[i][1] += b->xcm[1] - ybox*yprd;
      x[i][2] += b->xcm[2] - zbox*zprd;
    } else {
      x[i][0] += b->xcm[0] - xbox*xprd - ybox*xy - zbox*xz;
      x[i][1] += b->xcm[1] - ybox*yprd - zbox*yz;
      x[i][2] += b->xcm[2] - zbox*zprd;
    }

    // virial = unwrapped coords dotted into body constraint force
    // body constraint force = implied force due to v change minus f external
    // assume f does not include forces internal to body
    // 1/2 factor b/c final_integrate contributes other half
    // assume per-atom contribution is due to constraint force on that atom

    if (evflag) {
      if (rmass) massone = rmass[i];
      else massone = mass[type[i]];
      fc0 = massone*(v[i][0] - v0)/dtf - f[i][0];
      fc1 = massone*(v[i][1] - v1)/dtf - f[i][1];
      fc2 = massone*(v[i][2] - v2)/dtf - f[i][2];

      vr[0] = 0.5*x0*fc0;
      vr[1] = 0.5*x1*fc1;
      vr[2] = 0.5*x2*fc2;
      vr[3] = 0.5*x0*fc1;
      vr[4] = 0.5*x0*fc2;
      vr[5] = 0.5*x1*fc2;

      double rlist[1][3] = {{x0, x1, x2}};
      double flist[1][3] = {{0.5*fc0, 0.5*fc1, 0.5*fc2}};
      v_tally(1,&i,1.0,vr,rlist,flist,b->xgc);
    }
  }

  // update the position of geometric center
  for (int ibody = 0; ibody < nlocal_body + nghost_body; ibody++) {
    Body *b = &body[ibody];
    MathExtra::matvec(b->ex_space,b->ey_space,b->ez_space,
                      b->xgc_body,b->xgc);
    b->xgc[0] += b->xcm[0];
    b->xgc[1] += b->xcm[1];
    b->xgc[2] += b->xcm[2];
  }

  // set orientation, omega, angmom of each extended particle

  if (extended) {
    double theta_body,theta;
    double *shape,*quatatom,*inertiaatom;

    AtomVecEllipsoid::Bonus *ebonus;
    if (avec_ellipsoid) ebonus = avec_ellipsoid->bonus;
    AtomVecLine::Bonus *lbonus;
    if (avec_line) lbonus = avec_line->bonus;
    AtomVecTri::Bonus *tbonus;
    if (avec_tri) tbonus = avec_tri->bonus;
    double **omega = atom->omega;
    double **angmom = atom->angmom;
    double **mu = atom->mu;
    int *ellipsoid = atom->ellipsoid;
    int *line = atom->line;
    int *tri = atom->tri;

    for (int i = 0; i < nlocal; i++) {
      if (atom2body[i] < 0) continue;
      Body *b = &body[atom2body[i]];

      if (eflags[i] & SPHERE) {
        omega[i][0] = b->omega[0];
        omega[i][1] = b->omega[1];
        omega[i][2] = b->omega[2];
      } else if (eflags[i] & ELLIPSOID) {
        shape = ebonus[ellipsoid[i]].shape;
        quatatom = ebonus[ellipsoid[i]].quat;
        MathExtra::quatquat(b->quat,orient[i],quatatom);
        MathExtra::qnormalize(quatatom);
        ione[0] = EINERTIA*rmass[i] * (shape[1]*shape[1] + shape[2]*shape[2]);
        ione[1] = EINERTIA*rmass[i] * (shape[0]*shape[0] + shape[2]*shape[2]);
        ione[2] = EINERTIA*rmass[i] * (shape[0]*shape[0] + shape[1]*shape[1]);
        MathExtra::q_to_exyz(quatatom,exone,eyone,ezone);
        MathExtra::omega_to_angmom(b->omega,exone,eyone,ezone,ione,angmom[i]);
      } else if (eflags[i] & LINE) {
        if (b->quat[3] >= 0.0) theta_body = 2.0*acos(b->quat[0]);
        else theta_body = -2.0*acos(b->quat[0]);
        theta = orient[i][0] + theta_body;
        while (theta <= -MY_PI) theta += MY_2PI;
        while (theta > MY_PI) theta -= MY_2PI;
        lbonus[line[i]].theta = theta;
        omega[i][0] = b->omega[0];
        omega[i][1] = b->omega[1];
        omega[i][2] = b->omega[2];
      } else if (eflags[i] & TRIANGLE) {
        inertiaatom = tbonus[tri[i]].inertia;
        quatatom = tbonus[tri[i]].quat;
        MathExtra::quatquat(b->quat,orient[i],quatatom);
        MathExtra::qnormalize(quatatom);
        MathExtra::q_to_exyz(quatatom,exone,eyone,ezone);
        MathExtra::omega_to_angmom(b->omega,exone,eyone,ezone,
                                   inertiaatom,angmom[i]);
      }
      if (eflags[i] & DIPOLE) {
        MathExtra::quat_to_mat(b->quat,p);
        MathExtra::matvec(p,dorient[i],mu[i]);
        MathExtra::snormalize3(mu[i][3],mu[i],mu[i]);
      }
    }
  }
}

/* ----------------------------------------------------------------------
   set space-frame velocity of each atom in a rigid body
   set omega and angmom of extended particles
   v = Vcm + (W cross (x - Xcm))
------------------------------------------------------------------------- */

void FixRigidShell::set_v()
{
  int xbox,ybox,zbox;
  double x0,x1,x2,v0,v1,v2,fc0,fc1,fc2,massone;
  double ione[3],exone[3],eyone[3],ezone[3],delta[3],vr[6];

  double xprd = domain->xprd;
  double yprd = domain->yprd;
  double zprd = domain->zprd;
  double xy = domain->xy;
  double xz = domain->xz;
  double yz = domain->yz;

  double **x = atom->x;
  double **v = atom->v;
  double **f = atom->f;
  double *rmass = atom->rmass;
  double *mass = atom->mass;
  int *type = atom->type;
  int nlocal = atom->nlocal;

  // set v of each atom

  for (int i = 0; i < nlocal; i++) {
    if (atom2body[i] < 0) continue;
    Body *b = &body[atom2body[i]];

    MathExtra::matvec(b->ex_space,b->ey_space,b->ez_space,displace[i],delta);

    // save old velocities for virial

    if (evflag) {
      v0 = v[i][0];
      v1 = v[i][1];
      v2 = v[i][2];
    }

    v[i][0] = b->omega[1]*delta[2] - b->omega[2]*delta[1] + b->vcm[0];
    v[i][1] = b->omega[2]*delta[0] - b->omega[0]*delta[2] + b->vcm[1];
    v[i][2] = b->omega[0]*delta[1] - b->omega[1]*delta[0] + b->vcm[2];

    // virial = unwrapped coords dotted into body constraint force
    // body constraint force = implied force due to v change minus f external
    // assume f does not include forces internal to body
    // 1/2 factor b/c initial_integrate contributes other half
    // assume per-atom contribution is due to constraint force on that atom

    if (evflag) {
      if (rmass) massone = rmass[i];
      else massone = mass[type[i]];
      fc0 = massone*(v[i][0] - v0)/dtf - f[i][0];
      fc1 = massone*(v[i][1] - v1)/dtf - f[i][1];
      fc2 = massone*(v[i][2] - v2)/dtf - f[i][2];

      xbox = (xcmimage[i] & IMGMASK) - IMGMAX;
      ybox = (xcmimage[i] >> IMGBITS & IMGMASK) - IMGMAX;
      zbox = (xcmimage[i] >> IMG2BITS) - IMGMAX;

      if (triclinic == 0) {
        x0 = x[i][0] + xbox*xprd;
        x1 = x[i][1] + ybox*yprd;
        x2 = x[i][2] + zbox*zprd;
      } else {
        x0 = x[i][0] + xbox*xprd + ybox*xy + zbox*xz;
        x1 = x[i][1] + ybox*yprd + zbox*yz;
        x2 = x[i][2] + zbox*zprd;
      }

      vr[0] = 0.5*x0*fc0;
      vr[1] = 0.5*x1*fc1;
      vr[2] = 0.5*x2*fc2;
      vr[3] = 0.5*x0*fc1;
      vr[4] = 0.5*x0*fc2;
      vr[5] = 0.5*x1*fc2;

      double rlist[1][3] = {{x0, x1, x2}};
      double flist[1][3] = {{0.5*fc0, 0.5*fc1, 0.5*fc2}};
      v_tally(1,&i,1.0,vr,rlist,flist,b->xgc);
    }
  }

  // set omega, angmom of each extended particle

  if (extended) {
    double *shape,*quatatom,*inertiaatom;

    AtomVecEllipsoid::Bonus *ebonus;
    if (avec_ellipsoid) ebonus = avec_ellipsoid->bonus;
    AtomVecTri::Bonus *tbonus;
    if (avec_tri) tbonus = avec_tri->bonus;
    double **omega = atom->omega;
    double **angmom = atom->angmom;
    int *ellipsoid = atom->ellipsoid;
    int *tri = atom->tri;

    for (int i = 0; i < nlocal; i++) {
      if (atom2body[i] < 0) continue;
      Body *b = &body[atom2body[i]];

      if (eflags[i] & SPHERE) {
        omega[i][0] = b->omega[0];
        omega[i][1] = b->omega[1];
        omega[i][2] = b->omega[2];
      } else if (eflags[i] & ELLIPSOID) {
        shape = ebonus[ellipsoid[i]].shape;
        quatatom = ebonus[ellipsoid[i]].quat;
        ione[0] = EINERTIA*rmass[i] * (shape[1]*shape[1] + shape[2]*shape[2]);
        ione[1] = EINERTIA*rmass[i] * (shape[0]*shape[0] + shape[2]*shape[2]);
        ione[2] = EINERTIA*rmass[i] * (shape[0]*shape[0] + shape[1]*shape[1]);
        MathExtra::q_to_exyz(quatatom,exone,eyone,ezone);
        MathExtra::omega_to_angmom(b->omega,exone,eyone,ezone,ione,
                                   angmom[i]);
      } else if (eflags[i] & LINE) {
        omega[i][0] = b->omega[0];
        omega[i][1] = b->omega[1];
        omega[i][2] = b->omega[2];
      } else if (eflags[i] & TRIANGLE) {
        inertiaatom = tbonus[tri[i]].inertia;
        quatatom = tbonus[tri[i]].quat;
        MathExtra::q_to_exyz(quatatom,exone,eyone,ezone);
        MathExtra::omega_to_angmom(b->omega,exone,eyone,ezone,
                                   inertiaatom,angmom[i]);
      }
    }
  }
}

/* ----------------------------------------------------------------------
   one-time identification of which atoms are in which rigid bodies
   set bodytag for all owned atoms
------------------------------------------------------------------------- */

void FixRigidShell::create_bodies(tagint *bodyID)
{
  int i,m;

  // allocate buffer for input to rendezvous comm
  // ncount = # of my atoms in bodies

  int *mask = atom->mask;
  int nlocal = atom->nlocal;

  int ncount = 0;
  for (i = 0; i < nlocal; i++)
    if (mask[i] & groupbit) ncount++;

  int *proclist;
  memory->create(proclist,ncount,"rigid/shell:proclist");
  auto inbuf = (InRvous *) memory->smalloc(ncount*sizeof(InRvous),"rigid/shell:inbuf");

  // setup buf to pass to rendezvous comm
  // one BodyMsg datum for each constituent atom
  // datum = me, local index of atom, atomID, bodyID, unwrapped coords
  // owning proc for each datum = random hash of bodyID

  double **x = atom->x;
  tagint *tag = atom->tag;
  imageint *image = atom->image;

  m = 0;
  for (i = 0; i < nlocal; i++) {
    if (!(mask[i] & groupbit)) continue;
    proclist[m] = hashlittle(&bodyID[i],sizeof(tagint),0) % nprocs;
    inbuf[m].me = me;
    inbuf[m].ilocal = i;
    inbuf[m].atomID = tag[i];
    inbuf[m].bodyID = bodyID[i];
    domain->unmap(x[i],image[i],inbuf[m].x);
    m++;
  }

  // perform rendezvous operation
  // each proc owns random subset of bodies
  // receives all atoms in those bodies
  // func = compute bbox of each body, find atom closest to geometric center

  char *buf;
  int nreturn = comm->rendezvous(RVOUS,ncount,(char *) inbuf,sizeof(InRvous),
                                 0,proclist,
                                 rendezvous_body,0,buf,sizeof(OutRvous),
                                 (void *) this);
  auto outbuf = (OutRvous *) buf;

  memory->destroy(proclist);
  memory->sfree(inbuf);

  // set bodytag of all owned atoms based on outbuf info for constituent atoms

  for (i = 0; i < nlocal; i++)
    if (!(mask[i] & groupbit)) bodytag[i] = 0;

  for (m = 0; m < nreturn; m++)
    bodytag[outbuf[m].ilocal] = outbuf[m].atomID;

  memory->sfree(outbuf);

  // maxextent = max of rsqfar across all procs
  // if defined, include molecule->maxextent

  MPI_Allreduce(&rsqfar,&maxextent,1,MPI_DOUBLE,MPI_MAX,world);
  maxextent = sqrt(maxextent);
  if (onemols) {
    for (i = 0; i < nmol; i++)
      maxextent = MAX(maxextent,onemols[i]->maxextent);
  }
}

/* ----------------------------------------------------------------------
   process rigid bodies assigned to me
   buf = list of N BodyMsg datums
------------------------------------------------------------------------- */

int FixRigidShell::rendezvous_body(int n, char *inbuf,
                                   int &rflag, int *&proclist, char *&outbuf,
                                   void *ptr)
{
  int i,m;
  double delx,dely,delz,rsq;
  int *iclose;
  tagint *idclose;
  double *x,*xown,*rsqclose;
  double **bbox,**ctr;

  auto frsptr = (FixRigidShell *) ptr;
  Memory *memory = frsptr->memory;
  Error *error = frsptr->error;
  MPI_Comm world = frsptr->world;

  // setup hash
  // use STL map instead of atom->map
  //   b/c know nothing about body ID values specified by user
  // ncount = number of bodies assigned to me
  // key = body ID
  // value = index into Ncount-length data structure

  auto in = (InRvous *) inbuf;
  std::map<tagint,int> hash;
  tagint id;

  int ncount = 0;
  for (i = 0; i < n; i++) {
    id = in[i].bodyID;
    if (hash.find(id) == hash.end()) hash[id] = ncount++;
  }

  // bbox = bounding box of each rigid body

  memory->create(bbox,ncount,6,"rigid/shell:bbox");

  for (m = 0; m < ncount; m++) {
    bbox[m][0] = bbox[m][2] = bbox[m][4] = BIG;
    bbox[m][1] = bbox[m][3] = bbox[m][5] = -BIG;
  }

  for (i = 0; i < n; i++) {
    m = hash.find(in[i].bodyID)->second;
    x = in[i].x;
    bbox[m][0] = MIN(bbox[m][0],x[0]);
    bbox[m][1] = MAX(bbox[m][1],x[0]);
    bbox[m][2] = MIN(bbox[m][2],x[1]);
    bbox[m][3] = MAX(bbox[m][3],x[1]);
    bbox[m][4] = MIN(bbox[m][4],x[2]);
    bbox[m][5] = MAX(bbox[m][5],x[2]);
  }

  // check if any bbox is size 0.0, meaning rigid body is a single particle

  int flag = 0;
  for (m = 0; m < ncount; m++)
    if (bbox[m][0] == bbox[m][1] && bbox[m][2] == bbox[m][3] &&
        bbox[m][4] == bbox[m][5]) flag = 1;
  int flagall;
  MPI_Allreduce(&flag,&flagall,1,MPI_INT,MPI_SUM,world);    // sync here?
  if (flagall)
    error->all(FLERR,"One or more rigid bodies are a single particle");

  // ctr = geometric center pt of each rigid body

  memory->create(ctr,ncount,3,"rigid/shell:bbox");

  for (m = 0; m < ncount; m++) {
    ctr[m][0] = 0.5 * (bbox[m][0] + bbox[m][1]);
    ctr[m][1] = 0.5 * (bbox[m][2] + bbox[m][3]);
    ctr[m][2] = 0.5 * (bbox[m][4] + bbox[m][5]);
  }

  // idclose = atomID closest to center point of each body

  memory->create(idclose,ncount,"rigid/shell:idclose");
  memory->create(iclose,ncount,"rigid/shell:iclose");
  memory->create(rsqclose,ncount,"rigid/shell:rsqclose");
  for (m = 0; m < ncount; m++) rsqclose[m] = BIG;

  for (i = 0; i < n; i++) {
    m = hash.find(in[i].bodyID)->second;
    x = in[i].x;
    delx = x[0] - ctr[m][0];
    dely = x[1] - ctr[m][1];
    delz = x[2] - ctr[m][2];
    rsq = delx*delx + dely*dely + delz*delz;
    if (rsq <= rsqclose[m]) {
      if (rsq == rsqclose[m] && in[i].atomID > idclose[m]) continue;
      iclose[m] = i;
      idclose[m] = in[i].atomID;
      rsqclose[m] = rsq;
    }
  }

  // compute rsqfar for all bodies I own
  // set rsqfar back in caller

  double rsqfar = 0.0;

  for (i = 0; i < n; i++) {
    m = hash.find(in[i].bodyID)->second;
    xown = in[iclose[m]].x;
    x = in[i].x;
    delx = x[0] - xown[0];
    dely = x[1] - xown[1];
    delz = x[2] - xown[2];
    rsq = delx*delx + dely*dely + delz*delz;
    rsqfar = MAX(rsqfar,rsq);
  }

  frsptr->rsqfar = rsqfar;

  // pass list of OutRvous datums back to comm->rendezvous

  int nout = n;
  memory->create(proclist,nout,"rigid/shell:proclist");
  auto out = (OutRvous *) memory->smalloc(nout*sizeof(OutRvous),"rigid/shell:out");

  for (i = 0; i < nout; i++) {
    proclist[i] = in[i].me;
    out[i].ilocal = in[i].ilocal;
    m = hash.find(in[i].bodyID)->second;
    out[i].atomID = idclose[m];
  }

  outbuf = (char *) out;

  // clean up
  // Comm::rendezvous will delete proclist and out (outbuf)

  memory->destroy(bbox);
  memory->destroy(ctr);
  memory->destroy(idclose);
  memory->destroy(iclose);
  memory->destroy(rsqclose);

  // flag = 2: new outbuf

  rflag = 2;
  return nout;
}

/* ----------------------------------------------------------------------
   one-time initialization of rigid body attributes
   sets extended flags, masstotal, center-of-mass
   sets Cartesian and diagonalized inertia tensor
   sets body image flags
   may read some properties from inpfile
------------------------------------------------------------------------- */

void FixRigidShell::setup_bodies_static()
{
  int i,ibody;

  // extended = 1 if any particle in a rigid body is finite size
  //              or has a dipole moment

  extended = orientflag = dorientflag = 0;

  AtomVecEllipsoid::Bonus *ebonus;
  if (avec_ellipsoid) ebonus = avec_ellipsoid->bonus;
  AtomVecLine::Bonus *lbonus;
  if (avec_line) lbonus = avec_line->bonus;
  AtomVecTri::Bonus *tbonus;
  if (avec_tri) tbonus = avec_tri->bonus;
  double **mu = atom->mu;
  double *radius = atom->radius;
  double *rmass = atom->rmass;
  double *mass = atom->mass;
  int *ellipsoid = atom->ellipsoid;
  int *line = atom->line;
  int *tri = atom->tri;
  int *type = atom->type;
  int nlocal = atom->nlocal;

  if (atom->radius_flag || atom->ellipsoid_flag || atom->line_flag ||
      atom->tri_flag || atom->mu_flag) {
    int flag = 0;
    for (i = 0; i < nlocal; i++) {
      if (bodytag[i] == 0) continue;
      if (radius && radius[i] > 0.0) flag = 1;
      if (ellipsoid && ellipsoid[i] >= 0) flag = 1;
      if (line && line[i] >= 0) flag = 1;
      if (tri && tri[i] >= 0) flag = 1;
      if (mu && mu[i][3] > 0.0) flag = 1;
    }

    MPI_Allreduce(&flag,&extended,1,MPI_INT,MPI_MAX,world);
  }

  // extended = 1 if using molecule template with finite-size particles
  // require all molecules in template to have consistent radiusflag

  if (onemols) {
    int radiusflag = onemols[0]->radiusflag;
    for (i = 1; i < nmol; i++) {
      if (onemols[i]->radiusflag != radiusflag)
        error->all(FLERR,"Inconsistent use of finite-size particles "
                   "by molecule template molecules");
    }
    if (radiusflag) extended = 1;
  }

  // grow extended arrays and set extended flags for each particle
  // orientflag = 4 if any particle stores ellipsoid or tri orientation
  // orientflag = 1 if any particle stores line orientation
  // dorientflag = 1 if any particle stores dipole orientation

  if (extended) {
    if (atom->ellipsoid_flag) orientflag = 4;
    if (atom->line_flag) orientflag = 1;
    if (atom->tri_flag) orientflag = 4;
    if (atom->mu_flag) dorientflag = 1;
    grow_arrays(atom->nmax);

    for (i = 0; i < nlocal; i++) {
      eflags[i] = 0;
      if (bodytag[i] == 0) continue;

      // set to POINT or SPHERE or ELLIPSOID or LINE

      if (radius && radius[i] > 0.0) {
        eflags[i] |= SPHERE;
        eflags[i] |= OMEGA;
        eflags[i] |= TORQUE;
      } else if (ellipsoid && ellipsoid[i] >= 0) {
        eflags[i] |= ELLIPSOID;
        eflags[i] |= ANGMOM;
        eflags[i] |= TORQUE;
      } else if (line && line[i] >= 0) {
        eflags[i] |= LINE;
        eflags[i] |= OMEGA;
        eflags[i] |= TORQUE;
      } else if (tri && tri[i] >= 0) {
        eflags[i] |= TRIANGLE;
        eflags[i] |= ANGMOM;
        eflags[i] |= TORQUE;
      } else eflags[i] |= POINT;

      // set DIPOLE if atom->mu and mu[3] > 0.0

      if (atom->mu_flag && mu[i][3] > 0.0)
        eflags[i] |= DIPOLE;
    }
  }

  // set body xcmimage flags = true image flags

  imageint *image = atom->image;
  for (i = 0; i < nlocal; i++)
    if (bodytag[i] >= 0) xcmimage[i] = image[i];
    else xcmimage[i] = 0;

  // acquire ghost bodies via forward comm
  // set atom2body for ghost atoms via forward comm
  // set atom2body for other owned atoms via reset_atom2body()

  nghost_body = 0;
  commflag = FULL_BODY;
  comm->forward_comm(this);
  reset_atom2body();
  reset_angle2body();

  // compute mass & center-of-mass of each rigid body

  double **x = atom->x;

  double *xcm;
  double *xgc;

  for (ibody = 0; ibody < nlocal_body+nghost_body; ibody++) {
    xcm = body[ibody].xcm;
    xgc = body[ibody].xgc;
    xcm[0] = xcm[1] = xcm[2] = 0.0;
    xgc[0] = xgc[1] = xgc[2] = 0.0;
    body[ibody].mass = 0.0;
    body[ibody].volume = 0;
    body[ibody].density = 0;
    body[ibody].natoms = 0;
    body[ibody].nangles = 0;
  }

  // define unwraped coordinates 1,2,3 to store the unwrapped coordinates of each atom (i1,i2,i3) in each angle
  double unwrap[3], unwrap1[3], unwrap2[3], unwrap3[3];
  int i1, i2, i3;
  double massone;
  
  int nanglelist = neighbor->nanglelist;
  int **anglelist = neighbor->anglelist;


  // Cycling through each angle (not sure if this is a local or global list) and summing their Volume and COM to the respective body
  for (int n = 0; n < nanglelist; n++) {
      if (angle2body[n] < 0) continue;
      Body *b = &body[angle2body[n]];
    
      // Storing the three atoms in each angle
      i1 = anglelist[n][0];
      i2 = anglelist[n][1];
      i3 = anglelist[n][2];

      domain->unmap(x[i1],xcmimage[i1],unwrap1);
      domain->unmap(x[i2],xcmimage[i2],unwrap2);
      domain->unmap(x[i3],xcmimage[i3],unwrap3);

      xcm = b->xcm;
      xgc = b->xgc;

      b->volume += ((((unwrap2[1]-unwrap1[1])*(unwrap3[2]-unwrap1[2])) - ((unwrap3[1]-unwrap1[1])*(unwrap2[2]-unwrap1[2]))) *((unwrap1[0]+unwrap2[0]) + unwrap3[0]))/6.0;
      xcm[0] += ((((unwrap2[1]-unwrap1[1])*(unwrap3[2]-unwrap1[2])) - ((unwrap3[1]-unwrap1[1])*(unwrap2[2]-unwrap1[2]))) *(((unwrap1[0]*unwrap1[0])+unwrap2[0]*(unwrap1[0]+unwrap2[0]))+unwrap3[0]*((unwrap1[0]+unwrap2[0]) + unwrap3[0])))/24.0;
      xcm[1] += ((((unwrap3[0]-unwrap1[0])*(unwrap2[2]-unwrap1[2])) - ((unwrap2[0]-unwrap1[0])*(unwrap3[2]-unwrap1[2]))) *(((unwrap1[1]*unwrap1[1])+unwrap2[1]*(unwrap1[1]+unwrap2[1]))+unwrap3[1]*((unwrap1[1]+unwrap2[1]) + unwrap3[1])))/24.0;
      xcm[2] += ((((unwrap2[0]-unwrap1[0])*(unwrap3[1]-unwrap1[1])) - ((unwrap3[0]-unwrap1[0])*(unwrap2[1]-unwrap1[1]))) *(((unwrap1[2]*unwrap1[2])+unwrap2[2]*(unwrap1[2]+unwrap2[2]))+unwrap3[2]*((unwrap1[2]+unwrap2[2]) + unwrap3[2])))/24.0;
      xgc[0] += ((((unwrap2[1]-unwrap1[1])*(unwrap3[2]-unwrap1[2])) - ((unwrap3[1]-unwrap1[1])*(unwrap2[2]-unwrap1[2]))) *(((unwrap1[0]*unwrap1[0])+unwrap2[0]*(unwrap1[0]+unwrap2[0]))+unwrap3[0]*((unwrap1[0]+unwrap2[0]) + unwrap3[0])))/24.0;
      xgc[1] += ((((unwrap3[0]-unwrap1[0])*(unwrap2[2]-unwrap1[2])) - ((unwrap2[0]-unwrap1[0])*(unwrap3[2]-unwrap1[2]))) *(((unwrap1[1]*unwrap1[1])+unwrap2[1]*(unwrap1[1]+unwrap2[1]))+unwrap3[1]*((unwrap1[1]+unwrap2[1]) + unwrap3[1])))/24.0;
      xgc[2] += ((((unwrap2[0]-unwrap1[0])*(unwrap3[1]-unwrap1[1])) - ((unwrap3[0]-unwrap1[0])*(unwrap2[1]-unwrap1[1]))) *(((unwrap1[2]*unwrap1[2])+unwrap2[2]*(unwrap1[2]+unwrap2[2]))+unwrap3[2]*((unwrap1[2]+unwrap2[2]) + unwrap3[2])))/24.0;
      b->nangles++;
    }
    
  // Cycling through the local atoms and summing their mass to the respective body
  for (i = 0; i < nlocal; i++) {
    if (atom2body[i] < 0) continue;
    Body *b = &body[atom2body[i]];

    if (rmass) massone = rmass[i];
    else massone = mass[type[i]];
    // Still using the masses of the surface atoms to define the body mass - Eventuall update to be calculated from the tetrahedra
    // There will need to be some consistency between masses of surface atoms and body mass for contact models?
    b->mass += massone;
    b->natoms++;
  }

  // reverse communicate xcm, mass of all bodies

  commflag = XCM_MASS;
  comm->reverse_comm(this,8);

  for (ibody = 0; ibody < nlocal_body; ibody++) {
    xcm = body[ibody].xcm;
    xgc = body[ibody].xgc;
    
    // Setting each bodies' density
    body[ibody].density = body[ibody].mass/body[ibody].volume;
    xcm[0] /= body[ibody].volume;
    xcm[1] /= body[ibody].volume;
    xcm[2] /= body[ibody].volume;
    xgc[0] /= body[ibody].volume;
    xgc[1] /= body[ibody].volume;
    xgc[2] /= body[ibody].volume;
  }

  // set vcm, angmom = 0.0 in case inpfile is used
  // and doesn't overwrite all body's values
  // since setup_bodies_dynamic() will not be called

  double *vcm,*angmom;

  for (ibody = 0; ibody < nlocal_body; ibody++) {
    vcm = body[ibody].vcm;
    vcm[0] = vcm[1] = vcm[2] = 0.0;
    angmom = body[ibody].angmom;
    angmom[0] = angmom[1] = angmom[2] = 0.0;
  }

  // set rigid body image flags to default values

  for (ibody = 0; ibody < nlocal_body; ibody++)
    body[ibody].image = ((imageint) IMGMAX << IMG2BITS) |
      ((imageint) IMGMAX << IMGBITS) | IMGMAX;

  // overwrite masstotal, center-of-mass, image flags with file values
  // inbody[i] = 0/1 if Ith rigid body is initialized by file

  int *inbody;
  if (inpfile) {
    // must call it here so it doesn't override read in data but
    // initialize bodies whose dynamic settings not set in inpfile

    setup_bodies_dynamic();

    memory->create(inbody,nlocal_body,"rigid/shell:inbody");
    for (ibody = 0; ibody < nlocal_body; ibody++) inbody[ibody] = 0;
    readfile(0,nullptr,inbody);
  }

  // remap the xcm of each body back into simulation box
  //   and reset body and atom xcmimage flags via pre_neighbor()

  pre_neighbor();

  // compute 6 moments of inertia of each body in Cartesian reference frame
  // dx,dy,dz = coords relative to center-of-mass
  // symmetric 3x3 inertia tensor stored in Voigt notation as 6-vector

  memory->create(itensor,nlocal_body+nghost_body,6,"rigid/shell:itensor");
  for (ibody = 0; ibody < nlocal_body+nghost_body; ibody++)
    for (i = 0; i < 6; i++) itensor[ibody][i] = 0.0;

  double dx,dy,dz;
  double *inertia;

  for (int n = 0; n < nanglelist; n++) {
      if (angle2body[n] < 0) continue;
      Body *b = &body[angle2body[n]];
    
      // Storing the three atoms in each angle
      i1 = anglelist[n][0];
      i2 = anglelist[n][1];
      i3 = anglelist[n][2];

      domain->unmap(x[i1],xcmimage[i1],unwrap1);
      domain->unmap(x[i2],xcmimage[i2],unwrap2);
      domain->unmap(x[i3],xcmimage[i3],unwrap3);

      inertia = itensor[angle2body[n]];
      inertia[0] += (((unwrap2[1]-unwrap1[1])*(unwrap3[2]-unwrap1[2])) - ((unwrap3[1]-unwrap1[1])*(unwrap2[2]-unwrap1[2]))) *(unwrap1[0]*(unwrap1[0]*unwrap1[0])+unwrap2[0]*((unwrap1[0]*unwrap1[0])+unwrap2[0]*(unwrap1[0]+unwrap2[0]))+unwrap3[0]*(((unwrap1[0]*unwrap1[0])+unwrap2[0]*(unwrap1[0]+unwrap2[0]))+unwrap3[0]*((unwrap1[0]+unwrap2[0]) + unwrap3[0])));
      inertia[1] += (((unwrap3[0]-unwrap1[0])*(unwrap2[2]-unwrap1[2])) - ((unwrap2[0]-unwrap1[0])*(unwrap3[2]-unwrap1[2]))) *(unwrap1[1]*(unwrap1[1]*unwrap1[1])+unwrap2[1]*((unwrap1[1]*unwrap1[1])+unwrap2[1]*(unwrap1[1]+unwrap2[1]))+unwrap3[1]*(((unwrap1[1]*unwrap1[1])+unwrap2[1]*(unwrap1[1]+unwrap2[1]))+unwrap3[1]*((unwrap1[1]+unwrap2[1]) + unwrap3[1])));
      inertia[2] += (((unwrap2[0]-unwrap1[0])*(unwrap3[1]-unwrap1[1])) - ((unwrap3[0]-unwrap1[0])*(unwrap2[1]-unwrap1[1]))) *(unwrap1[2]*(unwrap1[2]*unwrap1[2])+unwrap2[2]*((unwrap1[2]*unwrap1[2])+unwrap2[2]*(unwrap1[2]+unwrap2[2]))+unwrap3[2]*(((unwrap1[2]*unwrap1[2])+unwrap2[2]*(unwrap1[2]+unwrap2[2]))+unwrap3[2]*((unwrap1[2]+unwrap2[2]) + unwrap3[2])));
      inertia[3] += (((unwrap2[1]-unwrap1[1])*(unwrap3[2]-unwrap1[2])) - ((unwrap3[1]-unwrap1[1])*(unwrap2[2]-unwrap1[2]))) * (unwrap1[1]*((((unwrap1[0]*unwrap1[0])+unwrap2[0]*(unwrap1[0]+unwrap2[0]))+unwrap3[0]*((unwrap1[0]+unwrap2[0]) + unwrap3[0]))+unwrap1[0]*(((unwrap1[0]+unwrap2[0]) + unwrap3[0])+unwrap1[0]))+unwrap2[1]*((((unwrap1[0]*unwrap1[0])+unwrap2[0]*(unwrap1[0]+unwrap2[0]))+unwrap3[0]*((unwrap1[0]+unwrap2[0]) + unwrap3[0]))+unwrap2[0]*(((unwrap1[0]+unwrap2[0]) + unwrap3[0])+unwrap2[0]))+unwrap3[1]*((((unwrap1[0]*unwrap1[0])+unwrap2[0]*(unwrap1[0]+unwrap2[0]))+unwrap3[0]*((unwrap1[0]+unwrap2[0]) + unwrap3[0]))+unwrap3[0]*(((unwrap1[0]+unwrap2[0]) + unwrap3[0])+unwrap3[0])));
      inertia[4] += (((unwrap3[0]-unwrap1[0])*(unwrap2[2]-unwrap1[2])) - ((unwrap2[0]-unwrap1[0])*(unwrap3[2]-unwrap1[2]))) * (unwrap1[2]*((((unwrap1[1]*unwrap1[1])+unwrap2[1]*(unwrap1[1]+unwrap2[1]))+unwrap3[1]*((unwrap1[1]+unwrap2[1]) + unwrap3[1]))+unwrap1[1]*(((unwrap1[1]+unwrap2[1]) + unwrap3[1])+unwrap1[1]))+unwrap2[2]*((((unwrap1[1]*unwrap1[1])+unwrap2[1]*(unwrap1[1]+unwrap2[1]))+unwrap3[1]*((unwrap1[1]+unwrap2[1]) + unwrap3[1]))+unwrap2[1]*(((unwrap1[1]+unwrap2[1]) + unwrap3[1])+unwrap2[1]))+unwrap3[2]*((((unwrap1[1]*unwrap1[1])+unwrap2[1]*(unwrap1[1]+unwrap2[1]))+unwrap3[1]*((unwrap1[1]+unwrap2[1]) + unwrap3[1]))+unwrap3[1]*(((unwrap1[1]+unwrap2[1]) + unwrap3[1])+unwrap3[1])));
      inertia[5] += (((unwrap2[0]-unwrap1[0])*(unwrap3[1]-unwrap1[1])) - ((unwrap3[0]-unwrap1[0])*(unwrap2[1]-unwrap1[1]))) * (unwrap1[0]*((((unwrap1[2]*unwrap1[2])+unwrap2[2]*(unwrap1[2]+unwrap2[2]))+unwrap3[2]*((unwrap1[2]+unwrap2[2]) + unwrap3[2]))+unwrap1[2]*(((unwrap1[2]+unwrap2[2]) + unwrap3[2])+unwrap1[2]))+unwrap2[0]*((((unwrap1[2]*unwrap1[2])+unwrap2[2]*(unwrap1[2]+unwrap2[2]))+unwrap3[2]*((unwrap1[2]+unwrap2[2]) + unwrap3[2]))+unwrap2[2]*(((unwrap1[2]+unwrap2[2]) + unwrap3[2])+unwrap2[2]))+unwrap3[0]*((((unwrap1[2]*unwrap1[2])+unwrap2[2]*(unwrap1[2]+unwrap2[2]))+unwrap3[2]*((unwrap1[2]+unwrap2[2]) + unwrap3[2]))+unwrap3[2]*(((unwrap1[2]+unwrap2[2]) + unwrap3[2])+unwrap3[2])));
    }

  // // extended particles may contribute extra terms to moments of inertia

  // if (extended) {
  //   double ivec[6];
  //   double *shape,*quatatom,*inertiaatom;
  //   double length,theta;

  //   for (i = 0; i < nlocal; i++) {
  //     if (atom2body[i] < 0) continue;
  //     inertia = itensor[atom2body[i]];

  //     if (rmass) massone = rmass[i];
  //     else massone = mass[type[i]];

  //     if (eflags[i] & SPHERE) {
  //       inertia[0] += SINERTIA*massone * radius[i]*radius[i];
  //       inertia[1] += SINERTIA*massone * radius[i]*radius[i];
  //       inertia[2] += SINERTIA*massone * radius[i]*radius[i];
  //     } else if (eflags[i] & ELLIPSOID) {
  //       shape = ebonus[ellipsoid[i]].shape;
  //       quatatom = ebonus[ellipsoid[i]].quat;
  //       MathExtra::inertia_ellipsoid(shape,quatatom,massone,ivec);
  //       inertia[0] += ivec[0];
  //       inertia[1] += ivec[1];
  //       inertia[2] += ivec[2];
  //       inertia[3] += ivec[3];
  //       inertia[4] += ivec[4];
  //       inertia[5] += ivec[5];
  //     } else if (eflags[i] & LINE) {
  //       length = lbonus[line[i]].length;
  //       theta = lbonus[line[i]].theta;
  //       MathExtra::inertia_line(length,theta,massone,ivec);
  //       inertia[0] += ivec[0];
  //       inertia[1] += ivec[1];
  //       inertia[2] += ivec[2];
  //       inertia[3] += ivec[3];
  //       inertia[4] += ivec[4];
  //       inertia[5] += ivec[5];
  //     } else if (eflags[i] & TRIANGLE) {
  //       inertiaatom = tbonus[tri[i]].inertia;
  //       quatatom = tbonus[tri[i]].quat;
  //       MathExtra::inertia_triangle(inertiaatom,quatatom,massone,ivec);
  //       inertia[0] += ivec[0];
  //       inertia[1] += ivec[1];
  //       inertia[2] += ivec[2];
  //       inertia[3] += ivec[3];
  //       inertia[4] += ivec[4];
  //       inertia[5] += ivec[5];
  //     }
  //   }
  // }

  // reverse communicate inertia tensor of all bodies

  commflag = ITENSOR;
  comm->reverse_comm(this,6);

  // overwrite Cartesian inertia tensor with file values

  if (inpfile) readfile(1,itensor,inbody);

  // diagonalize inertia tensor for each body via Jacobi rotations
  // inertia = 3 eigenvalues = principal moments of inertia
  // evectors and exzy_space = 3 evectors = principal axes of rigid body

  int ierror;
  double cross[3];
  double tensor[3][3],evectors[3][3];
  double *ex,*ey,*ez;

  for (ibody = 0; ibody < nlocal_body; ibody++) {

    tensor[0][0] = body[ibody].density * (((itensor[ibody][1] + itensor[ibody][2])/60.0) - body[ibody].volume*(body[ibody].xcm[1]*body[ibody].xcm[1] + body[ibody].xcm[2]*body[ibody].xcm[2]));
    tensor[1][1] = body[ibody].density * (((itensor[ibody][0] + itensor[ibody][2])/60.0) - body[ibody].volume*(body[ibody].xcm[2]*body[ibody].xcm[2] + body[ibody].xcm[0]*body[ibody].xcm[0]));
    tensor[2][2] = body[ibody].density * (((itensor[ibody][0] + itensor[ibody][1])/60.0) - body[ibody].volume*(body[ibody].xcm[0]*body[ibody].xcm[0] + body[ibody].xcm[1]*body[ibody].xcm[1]));
    
    tensor[0][1] = tensor[1][0] = -body[ibody].density * ((itensor[ibody][3]/120.0 - body[ibody].volume*body[ibody].xcm[0]*body[ibody].xcm[1]));
    tensor[1][2] = tensor[2][1] = -body[ibody].density * ((itensor[ibody][4]/120.0 - body[ibody].volume*body[ibody].xcm[1]*body[ibody].xcm[2]));  
    tensor[0][2] = tensor[2][0] = -body[ibody].density * ((itensor[ibody][5]/120.0 - body[ibody].volume*body[ibody].xcm[2]*body[ibody].xcm[0]));

    // std::cout << "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ BODY: " << ibody << ", Ixx 00: " << tensor[0][0]<< "~~~~~~~~~~~~~~~~~~~~~~~~~" << std::endl;
    // std::cout << "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ BODY: " << ibody << ", Iyy: 11 " << tensor[1][1]<< "~~~~~~~~~~~~~~~~~~~~~~~~~" << std::endl;
    // std::cout << "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ BODY: " << ibody << ", Izz: 22 " << tensor[2][2]<< "~~~~~~~~~~~~~~~~~~~~~~~~~" << std::endl;
    // std::cout << "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ BODY: " << ibody << ", Ixy: 12 21 " << tensor[1][2]<< "~~~~~~~~~~~~~~~~~~~~~~~~~" << std::endl;
    // std::cout << "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ BODY: " << ibody << ", Iyz: 02 20 " << tensor[0][2]<< "~~~~~~~~~~~~~~~~~~~~~~~~~" << std::endl;
    // std::cout << "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ BODY: " << ibody << ", Ixz: 01 10 " << tensor[0][1]<< "~~~~~~~~~~~~~~~~~~~~~~~~~" << std::endl;

    inertia = body[ibody].inertia;
    ierror = MathEigen::jacobi3(tensor,inertia,evectors);
    if (ierror) error->all(FLERR,
                           "Insufficient Jacobi rotations for rigid body");

    ex = body[ibody].ex_space;
    ex[0] = evectors[0][0];
    ex[1] = evectors[1][0];
    ex[2] = evectors[2][0];
    ey = body[ibody].ey_space;
    ey[0] = evectors[0][1];
    ey[1] = evectors[1][1];
    ey[2] = evectors[2][1];
    ez = body[ibody].ez_space;
    ez[0] = evectors[0][2];
    ez[1] = evectors[1][2];
    ez[2] = evectors[2][2];

    // if any principal moment < scaled EPSILON, set to 0.0

    double max;
    max = MAX(inertia[0],inertia[1]);
    max = MAX(max,inertia[2]);

    if (inertia[0] < EPSILON*max) inertia[0] = 0.0;
    if (inertia[1] < EPSILON*max) inertia[1] = 0.0;
    if (inertia[2] < EPSILON*max) inertia[2] = 0.0;

    std::cout << "~~~~~~~~~~~~~~~~~~~~~~ Body: " << ibody << ", Density: " << body[ibody].density << ", angles: " << body[ibody].nangles << ", Inertia: " << inertia[0] << ", " << inertia[1] << ", " << inertia[2] << ") ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~" << std::endl;
    // enforce 3 evectors as a right-handed coordinate system
    // flip 3rd vector if needed

    MathExtra::cross3(ex,ey,cross);
    if (MathExtra::dot3(cross,ez) < 0.0) MathExtra::negate3(ez);

    // create initial quaternion

    MathExtra::exyz_to_q(ex,ey,ez,body[ibody].quat);

    // convert geometric center position to principal axis coordinates
    // xcm is wrapped, but xgc is not initially
    xcm = body[ibody].xcm;
    xgc = body[ibody].xgc;
    double delta[3];
    MathExtra::sub3(xgc,xcm,delta);
    domain->minimum_image(delta);
    MathExtra::transpose_matvec(ex,ey,ez,delta,body[ibody].xgc_body);
    MathExtra::add3(xcm,delta,xgc);
  }

  // forward communicate updated info of all bodies

  commflag = INITIAL;
  comm->forward_comm(this,29);

  // displace = initial atom coords in basis of principal axes
  // set displace = 0.0 for atoms not in any rigid body
  // for extended particles, set their orientation wrt to rigid body

  double qc[4],delta[3];
  double *quatatom;
  double theta_body;

  for (i = 0; i < nlocal; i++) {
    if (atom2body[i] < 0) {
      displace[i][0] = displace[i][1] = displace[i][2] = 0.0;
      continue;
    }

    Body *b = &body[atom2body[i]];

    domain->unmap(x[i],xcmimage[i],unwrap);
    xcm = b->xcm;
    delta[0] = unwrap[0] - xcm[0];
    delta[1] = unwrap[1] - xcm[1];
    delta[2] = unwrap[2] - xcm[2];
    MathExtra::transpose_matvec(b->ex_space,b->ey_space,b->ez_space,
                                delta,displace[i]);

    if (extended) {
      if (eflags[i] & ELLIPSOID) {
        quatatom = ebonus[ellipsoid[i]].quat;
        MathExtra::qconjugate(b->quat,qc);
        MathExtra::quatquat(qc,quatatom,orient[i]);
        MathExtra::qnormalize(orient[i]);
      } else if (eflags[i] & LINE) {
        if (b->quat[3] >= 0.0) theta_body = 2.0*acos(b->quat[0]);
        else theta_body = -2.0*acos(b->quat[0]);
        orient[i][0] = lbonus[line[i]].theta - theta_body;
        while (orient[i][0] <= -MY_PI) orient[i][0] += MY_2PI;
        while (orient[i][0] > MY_PI) orient[i][0] -= MY_2PI;
        if (orientflag == 4) orient[i][1] = orient[i][2] = orient[i][3] = 0.0;
      } else if (eflags[i] & TRIANGLE) {
        quatatom = tbonus[tri[i]].quat;
        MathExtra::qconjugate(b->quat,qc);
        MathExtra::quatquat(qc,quatatom,orient[i]);
        MathExtra::qnormalize(orient[i]);
      } else if (orientflag == 4) {
        orient[i][0] = orient[i][1] = orient[i][2] = orient[i][3] = 0.0;
      } else if (orientflag == 1)
        orient[i][0] = 0.0;

      if (eflags[i] & DIPOLE) {
        MathExtra::transpose_matvec(b->ex_space,b->ey_space,b->ez_space,
                                    mu[i],dorient[i]);
        MathExtra::snormalize3(mu[i][3],dorient[i],dorient[i]);
      } else if (dorientflag)
        dorient[i][0] = dorient[i][1] = dorient[i][2] = 0.0;
    }
  }

  // test for valid principal moments & axes
  // recompute moments of inertia around new axes
  // 3 diagonal moments should equal principal moments
  // 3 off-diagonal moments should be 0.0
  // extended particles may contribute extra terms to moments of inertia

  for (ibody = 0; ibody < nlocal_body+nghost_body; ibody++)
    for (i = 0; i < 6; i++) itensor[ibody][i] = 0.0;

  for (int n = 0; n < nanglelist; n++) {
    if (angle2body[n] < 0) continue;
    
    i1 = anglelist[n][0];
    i2 = anglelist[n][1];
    i3 = anglelist[n][2];
    
    inertia = itensor[angle2body[n]];

    inertia[0] += body[angle2body[n]].density * ((((((displace[i3][0]-displace[i1][0])*(displace[i2][2]-displace[i1][2])) - ((displace[i2][0]-displace[i1][0])*(displace[i3][2]-displace[i1][2]))) *(displace[i1][1]*(displace[i1][1]*displace[i1][1])+displace[i2][1]*((displace[i1][1]*displace[i1][1])+displace[i2][1]*(displace[i1][1]+displace[i2][1]))+displace[i3][1]*(((displace[i1][1]*displace[i1][1])+displace[i2][1]*(displace[i1][1]+displace[i2][1]))+displace[i3][1]*((displace[i1][1]+displace[i2][1]) + displace[i3][1]))) + (((displace[i2][0]-displace[i1][0])*(displace[i3][1]-displace[i1][1])) - ((displace[i3][0]-displace[i1][0])*(displace[i2][1]-displace[i1][1]))) *(displace[i1][2]*(displace[i1][2]*displace[i1][2])+displace[i2][2]*((displace[i1][2]*displace[i1][2])+displace[i2][2]*(displace[i1][2]+displace[i2][2]))+displace[i3][2]*(((displace[i1][2]*displace[i1][2])+displace[i2][2]*(displace[i1][2]+displace[i2][2]))+displace[i3][2]*((displace[i1][2]+displace[i2][2]) + displace[i3][2]))))/60.0));
    inertia[1] += body[angle2body[n]].density * ((((((displace[i2][1]-displace[i1][1])*(displace[i3][2]-displace[i1][2])) - ((displace[i3][1]-displace[i1][1])*(displace[i2][2]-displace[i1][2]))) *(displace[i1][0]*(displace[i1][0]*displace[i1][0])+displace[i2][0]*((displace[i1][0]*displace[i1][0])+displace[i2][0]*(displace[i1][0]+displace[i2][0]))+displace[i3][0]*(((displace[i1][0]*displace[i1][0])+displace[i2][0]*(displace[i1][0]+displace[i2][0]))+displace[i3][0]*((displace[i1][0]+displace[i2][0]) + displace[i3][0]))) + (((displace[i2][0]-displace[i1][0])*(displace[i3][1]-displace[i1][1])) - ((displace[i3][0]-displace[i1][0])*(displace[i2][1]-displace[i1][1]))) *(displace[i1][2]*(displace[i1][2]*displace[i1][2])+displace[i2][2]*((displace[i1][2]*displace[i1][2])+displace[i2][2]*(displace[i1][2]+displace[i2][2]))+displace[i3][2]*(((displace[i1][2]*displace[i1][2])+displace[i2][2]*(displace[i1][2]+displace[i2][2]))+displace[i3][2]*((displace[i1][2]+displace[i2][2]) + displace[i3][2]))))/60.0));
    inertia[2] += body[angle2body[n]].density * ((((((displace[i2][1]-displace[i1][1])*(displace[i3][2]-displace[i1][2])) - ((displace[i3][1]-displace[i1][1])*(displace[i2][2]-displace[i1][2]))) *(displace[i1][0]*(displace[i1][0]*displace[i1][0])+displace[i2][0]*((displace[i1][0]*displace[i1][0])+displace[i2][0]*(displace[i1][0]+displace[i2][0]))+displace[i3][0]*(((displace[i1][0]*displace[i1][0])+displace[i2][0]*(displace[i1][0]+displace[i2][0]))+displace[i3][0]*((displace[i1][0]+displace[i2][0]) + displace[i3][0]))) + (((displace[i3][0]-displace[i1][0])*(displace[i2][2]-displace[i1][2])) - ((displace[i2][0]-displace[i1][0])*(displace[i3][2]-displace[i1][2]))) *(displace[i1][1]*(displace[i1][1]*displace[i1][1])+displace[i2][1]*((displace[i1][1]*displace[i1][1])+displace[i2][1]*(displace[i1][1]+displace[i2][1]))+displace[i3][1]*(((displace[i1][1]*displace[i1][1])+displace[i2][1]*(displace[i1][1]+displace[i2][1]))+displace[i3][1]*((displace[i1][1]+displace[i2][1]) + displace[i3][1]))))/60.0));
    inertia[3] -= body[angle2body[n]].density * ((((((displace[i2][1]-displace[i1][1])*(displace[i3][2]-displace[i1][2])) - ((displace[i3][1]-displace[i1][1])*(displace[i2][2]-displace[i1][2]))) * (displace[i1][1]*((((displace[i1][0]*displace[i1][0])+displace[i2][0]*(displace[i1][0]+displace[i2][0]))+displace[i3][0]*((displace[i1][0]+displace[i2][0]) + displace[i3][0]))+displace[i1][0]*(((displace[i1][0]+displace[i2][0]) + displace[i3][0])+displace[i1][0]))+displace[i2][1]*((((displace[i1][0]*displace[i1][0])+displace[i2][0]*(displace[i1][0]+displace[i2][0]))+displace[i3][0]*((displace[i1][0]+displace[i2][0]) + displace[i3][0]))+displace[i2][0]*(((displace[i1][0]+displace[i2][0]) + displace[i3][0])+displace[i2][0]))+displace[i3][1]*((((displace[i1][0]*displace[i1][0])+displace[i2][0]*(displace[i1][0]+displace[i2][0]))+displace[i3][0]*((displace[i1][0]+displace[i2][0]) + displace[i3][0]))+displace[i3][0]*(((displace[i1][0]+displace[i2][0]) + displace[i3][0])+displace[i3][0]))))/120.0));
    inertia[4] -= body[angle2body[n]].density * ((((((displace[i3][0]-displace[i1][0])*(displace[i2][2]-displace[i1][2])) - ((displace[i2][0]-displace[i1][0])*(displace[i3][2]-displace[i1][2]))) * (displace[i1][2]*((((displace[i1][1]*displace[i1][1])+displace[i2][1]*(displace[i1][1]+displace[i2][1]))+displace[i3][1]*((displace[i1][1]+displace[i2][1]) + displace[i3][1]))+displace[i1][1]*(((displace[i1][1]+displace[i2][1]) + displace[i3][1])+displace[i1][1]))+displace[i2][2]*((((displace[i1][1]*displace[i1][1])+displace[i2][1]*(displace[i1][1]+displace[i2][1]))+displace[i3][1]*((displace[i1][1]+displace[i2][1]) + displace[i3][1]))+displace[i2][1]*(((displace[i1][1]+displace[i2][1]) + displace[i3][1])+displace[i2][1]))+displace[i3][2]*((((displace[i1][1]*displace[i1][1])+displace[i2][1]*(displace[i1][1]+displace[i2][1]))+displace[i3][1]*((displace[i1][1]+displace[i2][1]) + displace[i3][1]))+displace[i3][1]*(((displace[i1][1]+displace[i2][1]) + displace[i3][1])+displace[i3][1]))))/120.0));  
    inertia[5] -= body[angle2body[n]].density * ((((((displace[i2][0]-displace[i1][0])*(displace[i3][1]-displace[i1][1])) - ((displace[i3][0]-displace[i1][0])*(displace[i2][1]-displace[i1][1]))) * (displace[i1][0]*((((displace[i1][2]*displace[i1][2])+displace[i2][2]*(displace[i1][2]+displace[i2][2]))+displace[i3][2]*((displace[i1][2]+displace[i2][2]) + displace[i3][2]))+displace[i1][2]*(((displace[i1][2]+displace[i2][2]) + displace[i3][2])+displace[i1][2]))+displace[i2][0]*((((displace[i1][2]*displace[i1][2])+displace[i2][2]*(displace[i1][2]+displace[i2][2]))+displace[i3][2]*((displace[i1][2]+displace[i2][2]) + displace[i3][2]))+displace[i2][2]*(((displace[i1][2]+displace[i2][2]) + displace[i3][2])+displace[i2][2]))+displace[i3][0]*((((displace[i1][2]*displace[i1][2])+displace[i2][2]*(displace[i1][2]+displace[i2][2]))+displace[i3][2]*((displace[i1][2]+displace[i2][2]) + displace[i3][2]))+displace[i3][2]*(((displace[i1][2]+displace[i2][2]) + displace[i3][2])+displace[i3][2]))))/120.0));
  }

  // if (extended) {
  //   double ivec[6];
  //   double *shape,*inertiaatom;
  //   double length;

  //   for (i = 0; i < nlocal; i++) {
  //     if (atom2body[i] < 0) continue;
  //     inertia = itensor[atom2body[i]];

  //     if (rmass) massone = rmass[i];
  //     else massone = mass[type[i]];

  //     if (eflags[i] & SPHERE) {
  //       inertia[0] += SINERTIA*massone * radius[i]*radius[i];
  //       inertia[1] += SINERTIA*massone * radius[i]*radius[i];
  //       inertia[2] += SINERTIA*massone * radius[i]*radius[i];
  //     } else if (eflags[i] & ELLIPSOID) {
  //       shape = ebonus[ellipsoid[i]].shape;
  //       MathExtra::inertia_ellipsoid(shape,orient[i],massone,ivec);
  //       inertia[0] += ivec[0];
  //       inertia[1] += ivec[1];
  //       inertia[2] += ivec[2];
  //       inertia[3] += ivec[3];
  //       inertia[4] += ivec[4];
  //       inertia[5] += ivec[5];
  //     } else if (eflags[i] & LINE) {
  //       length = lbonus[line[i]].length;
  //       MathExtra::inertia_line(length,orient[i][0],massone,ivec);
  //       inertia[0] += ivec[0];
  //       inertia[1] += ivec[1];
  //       inertia[2] += ivec[2];
  //       inertia[3] += ivec[3];
  //       inertia[4] += ivec[4];
  //       inertia[5] += ivec[5];
  //     } else if (eflags[i] & TRIANGLE) {
  //       inertiaatom = tbonus[tri[i]].inertia;
  //       MathExtra::inertia_triangle(inertiaatom,orient[i],massone,ivec);
  //       inertia[0] += ivec[0];
  //       inertia[1] += ivec[1];
  //       inertia[2] += ivec[2];
  //       inertia[3] += ivec[3];
  //       inertia[4] += ivec[4];
  //       inertia[5] += ivec[5];
  //     }
  //   }
  // }

  // reverse communicate inertia tensor of all bodies

  commflag = ITENSOR;
  comm->reverse_comm(this,6);

  // error check that re-computed moments of inertia match diagonalized ones
  // do not do test for bodies with params read from inpfile

  double norm;
  for (ibody = 0; ibody < nlocal_body; ibody++) {
    if (inpfile && inbody[ibody]) continue;
    inertia = body[ibody].inertia;

    if (inertia[0] == 0.0) {
      if (fabs(itensor[ibody][0]) > TOLERANCE)
        error->all(FLERR,"Fix rigid: Bad principal moments");
    } else {
      if (fabs((itensor[ibody][0]-inertia[0])/inertia[0]) >
          TOLERANCE) error->all(FLERR,"Fix rigid: Bad principal moments");
    }
    if (inertia[1] == 0.0) {
      if (fabs(itensor[ibody][1]) > TOLERANCE)
        error->all(FLERR,"Fix rigid: Bad principal moments");
    } else {
      if (fabs((itensor[ibody][1]-inertia[1])/inertia[1]) >
          TOLERANCE) error->all(FLERR,"Fix rigid: Bad principal moments");
    }
    if (inertia[2] == 0.0) {
      if (fabs(itensor[ibody][2]) > TOLERANCE)
        error->all(FLERR,"Fix rigid: Bad principal moments");
    } else {
      if (fabs((itensor[ibody][2]-inertia[2])/inertia[2]) >
          TOLERANCE) error->all(FLERR,"Fix rigid: Bad principal moments");
    }
    norm = (inertia[0] + inertia[1] + inertia[2]) / 3.0;
    if (fabs(itensor[ibody][3]/norm) > TOLERANCE ||
        fabs(itensor[ibody][4]/norm) > TOLERANCE ||
        fabs(itensor[ibody][5]/norm) > TOLERANCE)
      error->all(FLERR,"Fix rigid: Bad principal moments");
  }

  // clean up

  memory->destroy(itensor);
  if (inpfile) memory->destroy(inbody);
}

/* ----------------------------------------------------------------------
   one-time initialization of dynamic rigid body attributes
   vcm and angmom, computed explicitly from constituent particles
   not done if body properties read from file, e.g. for overlapping particles
------------------------------------------------------------------------- */

void FixRigidShell::setup_bodies_dynamic()
{
  int i,ibody;
  double massone,radone;

  // sum vcm, angmom across all rigid bodies
  // vcm = velocity of COM
  // angmom = angular momentum around COM

  double **x = atom->x;
  double **v = atom->v;
  double *rmass = atom->rmass;
  double *mass = atom->mass;
  int *type = atom->type;
  int nlocal = atom->nlocal;

  double *xcm,*vcm,*acm;
  double dx,dy,dz;
  double unwrap[3];

  for (ibody = 0; ibody < nlocal_body+nghost_body; ibody++) {
    vcm = body[ibody].vcm;
    vcm[0] = vcm[1] = vcm[2] = 0.0;
    acm = body[ibody].angmom;
    acm[0] = acm[1] = acm[2] = 0.0;
  }

  for (i = 0; i < nlocal; i++) {
    if (atom2body[i] < 0) continue;
    Body *b = &body[atom2body[i]];

    if (rmass) massone = rmass[i];
    else massone = mass[type[i]];

    vcm = b->vcm;
    vcm[0] += v[i][0] * massone;
    vcm[1] += v[i][1] * massone;
    vcm[2] += v[i][2] * massone;

    domain->unmap(x[i],xcmimage[i],unwrap);
    xcm = b->xcm;
    dx = unwrap[0] - xcm[0];
    dy = unwrap[1] - xcm[1];
    dz = unwrap[2] - xcm[2];

    acm = b->angmom;
    acm[0] += dy * massone*v[i][2] - dz * massone*v[i][1];
    acm[1] += dz * massone*v[i][0] - dx * massone*v[i][2];
    acm[2] += dx * massone*v[i][1] - dy * massone*v[i][0];
  }

  // extended particles add their rotation to angmom of body

  if (extended) {
    AtomVecLine::Bonus *lbonus;
    if (avec_line) lbonus = avec_line->bonus;
    double **omega = atom->omega;
    double **angmom = atom->angmom;
    double *radius = atom->radius;
    int *line = atom->line;

    for (i = 0; i < nlocal; i++) {
      if (atom2body[i] < 0) continue;
      Body *b = &body[atom2body[i]];

      if (eflags[i] & OMEGA) {
        if (eflags[i] & SPHERE) {
          radone = radius[i];
          acm = b->angmom;
          acm[0] += SINERTIA*rmass[i] * radone*radone * omega[i][0];
          acm[1] += SINERTIA*rmass[i] * radone*radone * omega[i][1];
          acm[2] += SINERTIA*rmass[i] * radone*radone * omega[i][2];
        } else if (eflags[i] & LINE) {
          radone = lbonus[line[i]].length;
          b->angmom[2] += LINERTIA*rmass[i] * radone*radone * omega[i][2];
        }
      }
      if (eflags[i] & ANGMOM) {
        acm = b->angmom;
        acm[0] += angmom[i][0];
        acm[1] += angmom[i][1];
        acm[2] += angmom[i][2];
      }
    }
  }

  // reverse communicate vcm, angmom of all bodies

  commflag = VCM_ANGMOM;
  comm->reverse_comm(this,6);

  // normalize velocity of COM

  for (ibody = 0; ibody < nlocal_body; ibody++) {
    vcm = body[ibody].vcm;
    vcm[0] /= body[ibody].mass;
    vcm[1] /= body[ibody].mass;
    vcm[2] /= body[ibody].mass;
  }
}

/* ----------------------------------------------------------------------
   read per rigid body info from user-provided file
   which = 0 to read everything except 6 moments of inertia
   which = 1 to read just 6 moments of inertia
   flag inbody = 0 for local bodies this proc initializes from file
   nlines = # of lines of rigid body info, 0 is OK
   one line = rigid-ID mass xcm ycm zcm ixx iyy izz ixy ixz iyz
              vxcm vycm vzcm lx ly lz
   where rigid-ID = mol-ID for fix rigid/shell
------------------------------------------------------------------------- */

void FixRigidShell::readfile(int which, double **array, int *inbody)
{
  int nchunk,eofflag,nlines,xbox,ybox,zbox;
  FILE *fp;
  char *eof,*start,*next,*buf;
  char line[MAXLINE];

  // create local hash with key/value pairs
  // key = mol ID of bodies my atoms own
  // value = index into local body array

  int nlocal = atom->nlocal;

  std::map<tagint,int> hash;
  for (int i = 0; i < nlocal; i++)
    if (bodyown[i] >= 0) hash[atom->molecule[i]] = bodyown[i];

  // open file and read header

  if (me == 0) {
    fp = fopen(inpfile,"r");
    if (fp == nullptr)
      error->one(FLERR,"Cannot open fix rigid/shell file {}: {}", inpfile, utils::getsyserror());
    while (true) {
      eof = fgets(line,MAXLINE,fp);
      if (eof == nullptr) error->one(FLERR,"Unexpected end of fix rigid/shell file");
      start = &line[strspn(line," \t\n\v\f\r")];
      if (*start != '\0' && *start != '#') break;
    }
    nlines = utils::inumeric(FLERR, utils::trim(line), true, lmp);
    if (which == 0)
      utils::logmesg(lmp, "Reading rigid body data for {} bodies from file {}\n", nlines, inpfile);
    if (nlines == 0) fclose(fp);
  }
  MPI_Bcast(&nlines,1,MPI_INT,0,world);

  // empty file with 0 lines is needed to trigger initial restart file
  // generation when no infile was previously used.

  if (nlines == 0) return;
  else if (nlines < 0) error->all(FLERR,"Fix rigid infile has incorrect format");

  auto buffer = new char[CHUNK*MAXLINE];
  int nread = 0;
  while (nread < nlines) {
    nchunk = MIN(nlines-nread,CHUNK);
    eofflag = utils::read_lines_from_file(fp,nchunk,MAXLINE,buffer,me,world);
    if (eofflag) error->all(FLERR,"Unexpected end of fix rigid/shell file");

    buf = buffer;
    next = strchr(buf,'\n');
    *next = '\0';
    int nwords = utils::count_words(utils::trim_comment(buf));
    *next = '\n';

    if (nwords != ATTRIBUTE_PERBODY)
      error->all(FLERR,"Incorrect rigid body format in fix rigid/shell file");

    // loop over lines of rigid body attributes
    // tokenize the line into values
    // id = rigid body ID = mol-ID
    // for which = 0, store all but inertia directly in body struct
    // for which = 1, store inertia tensor array, invert 3,4,5 values to Voigt

    for (int i = 0; i < nchunk; i++) {
      next = strchr(buf,'\n');
      *next = '\0';

      try {
        ValueTokenizer values(buf);
        tagint id = values.next_tagint();

        if (id <= 0 || id > maxmol)
          error->all(FLERR,"Invalid rigid body molecude ID {} in fix rigid/shell file", id);

        if (hash.find(id) == hash.end()) {
          buf = next + 1;
          continue;
        }
        int m = hash[id];
        inbody[m] = 1;

        if (which == 0) {
          body[m].mass = values.next_double();
          body[m].xcm[0] = values.next_double();
          body[m].xcm[1] = values.next_double();
          body[m].xcm[2] = values.next_double();
          values.skip(6);
          body[m].vcm[0] = values.next_double();
          body[m].vcm[1] = values.next_double();
          body[m].vcm[2] = values.next_double();
          body[m].angmom[0] = values.next_double();
          body[m].angmom[1] = values.next_double();
          body[m].angmom[2] = values.next_double();
          xbox = values.next_int();
          ybox = values.next_int();
          zbox = values.next_int();
          body[m].image = ((imageint) (xbox + IMGMAX) & IMGMASK) |
            (((imageint) (ybox + IMGMAX) & IMGMASK) << IMGBITS) |
            (((imageint) (zbox + IMGMAX) & IMGMASK) << IMG2BITS);
        } else {
          values.skip(4);
          array[m][0] = values.next_double();
          array[m][1] = values.next_double();
          array[m][2] = values.next_double();
          array[m][5] = values.next_double();
          array[m][4] = values.next_double();
          array[m][3] = values.next_double();
        }
      } catch (TokenizerException &e) {
        error->all(FLERR, "Invalid fix rigid/shell infile: {}", e.what());
      }
      buf = next + 1;
    }
    nread += nchunk;
  }

  if (me == 0) fclose(fp);
  delete[] buffer;
}

/* ----------------------------------------------------------------------
   write out restart info for mass, COM, inertia tensor to file
   identical format to inpfile option, so info can be read in when restarting
   each proc contributes info for rigid bodies it owns
------------------------------------------------------------------------- */

void FixRigidShell::write_restart_file(const char *file)
{
  FILE *fp;

  // do not write file if bodies have not yet been initialized

  if (!setupflag) return;

  // proc 0 opens file and writes header

  if (me == 0) {
    auto outfile = std::string(file) + ".rigid";
    fp = fopen(outfile.c_str(),"w");
    if (fp == nullptr)
      error->one(FLERR,"Cannot open fix rigid restart file {}: {}",outfile,utils::getsyserror());

    fmt::print(fp,"# fix rigid mass, COM, inertia tensor info for "
               "{} bodies on timestep {}\n\n",nbody,update->ntimestep);
    fmt::print(fp,"{}\n",nbody);
  }

  // communication buffer for all my rigid body info
  // max_size = largest buffer needed by any proc
  // ncol = # of values per line in output file

  int ncol = ATTRIBUTE_PERBODY;
  int sendrow = nlocal_body;
  int maxrow;
  MPI_Allreduce(&sendrow,&maxrow,1,MPI_INT,MPI_MAX,world);

  double **buf;
  if (me == 0) memory->create(buf,MAX(1,maxrow),ncol,"rigid/shell:buf");
  else memory->create(buf,MAX(1,sendrow),ncol,"rigid/shell:buf");

  // pack my rigid body info into buf
  // compute I tensor against xyz axes from diagonalized I and current quat
  // Ispace = P Idiag P_transpose
  // P is stored column-wise in exyz_space

  double p[3][3],pdiag[3][3],ispace[3][3];

  for (int i = 0; i < nlocal_body; i++) {
    MathExtra::col2mat(body[i].ex_space,body[i].ey_space,body[i].ez_space,p);
    MathExtra::times3_diag(p,body[i].inertia,pdiag);
    MathExtra::times3_transpose(pdiag,p,ispace);

    buf[i][0] = atom->molecule[body[i].ilocal];
    buf[i][1] = body[i].mass;
    buf[i][2] = body[i].xcm[0];
    buf[i][3] = body[i].xcm[1];
    buf[i][4] = body[i].xcm[2];
    buf[i][5] = ispace[0][0];
    buf[i][6] = ispace[1][1];
    buf[i][7] = ispace[2][2];
    buf[i][8] = ispace[0][1];
    buf[i][9] = ispace[0][2];
    buf[i][10] = ispace[1][2];
    buf[i][11] = body[i].vcm[0];
    buf[i][12] = body[i].vcm[1];
    buf[i][13] = body[i].vcm[2];
    buf[i][14] = body[i].angmom[0];
    buf[i][15] = body[i].angmom[1];
    buf[i][16] = body[i].angmom[2];
    buf[i][17] = (body[i].image & IMGMASK) - IMGMAX;
    buf[i][18] = (body[i].image >> IMGBITS & IMGMASK) - IMGMAX;
    buf[i][19] = (body[i].image >> IMG2BITS) - IMGMAX;
  }

  // write one chunk of rigid body info per proc to file
  // proc 0 pings each proc, receives its chunk, writes to file
  // all other procs wait for ping, send their chunk to proc 0

  int tmp,recvrow;

  if (me == 0) {
    MPI_Status status;
    MPI_Request request;
    for (int iproc = 0; iproc < nprocs; iproc++) {
      if (iproc) {
        MPI_Irecv(&buf[0][0],maxrow*ncol,MPI_DOUBLE,iproc,0,world,&request);
        MPI_Send(&tmp,0,MPI_INT,iproc,0,world);
        MPI_Wait(&request,&status);
        MPI_Get_count(&status,MPI_DOUBLE,&recvrow);
        recvrow /= ncol;
      } else recvrow = sendrow;

      for (int i = 0; i < recvrow; i++)
        fprintf(fp,"%d %-1.16e %-1.16e %-1.16e %-1.16e "
                "%-1.16e %-1.16e %-1.16e %-1.16e %-1.16e %-1.16e "
                "%-1.16e %-1.16e %-1.16e %-1.16e %-1.16e %-1.16e %d %d %d\n",
                static_cast<int> (buf[i][0]),buf[i][1],
                buf[i][2],buf[i][3],buf[i][4],
                buf[i][5],buf[i][6],buf[i][7],
                buf[i][8],buf[i][9],buf[i][10],
                buf[i][11],buf[i][12],buf[i][13],
                buf[i][14],buf[i][15],buf[i][16],
                static_cast<int> (buf[i][17]),
                static_cast<int> (buf[i][18]),
                static_cast<int> (buf[i][19]));
    }

  } else {
    MPI_Recv(&tmp,0,MPI_INT,0,0,world,MPI_STATUS_IGNORE);
    MPI_Rsend(&buf[0][0],sendrow*ncol,MPI_DOUBLE,0,0,world);
  }

  // clean up and close file

  memory->destroy(buf);
  if (me == 0) fclose(fp);
}

/* ----------------------------------------------------------------------
   allocate local atom-based arrays
------------------------------------------------------------------------- */

void FixRigidShell::grow_arrays(int nmax)
{
  memory->grow(bodyown,nmax,"rigid/shell:bodyown");
  memory->grow(bodytag,nmax,"rigid/shell:bodytag");
  memory->grow(atom2body,nmax,"rigid/shell:atom2body");
  memory->grow(angle2body,(neighbor->nanglelist),"rigid/shell:angle2body"); 
  // memory->grow(angle2body,(2*(nbody*nmax)-4*nbody),"rigid/shell:angle2body");  
  memory->grow(xcmimage,nmax,"rigid/shell:xcmimage");
  memory->grow(displace,nmax,3,"rigid/shell:displace");
  memory->grow(vertexdata,nmax,7,"rigid/shell:vertexdata");
  if (extended) {
    memory->grow(eflags,nmax,"rigid/shell:eflags");
    if (orientflag) memory->grow(orient,nmax,orientflag,"rigid/shell:orient");
    if (dorientflag) memory->grow(dorient,nmax,3,"rigid/shell:dorient");
  }

  // check for regrow of vatom
  // must be done whether per-atom virial is accumulated on this step or not
  //   b/c this is only time grow_array() may be called
  // need to regrow b/c vatom is calculated before and after atom migration

  if (nmax > maxvatom) {
    maxvatom = atom->nmax;
    memory->grow(vatom,maxvatom,6,"fix:vatom");
  }
}

/* ----------------------------------------------------------------------
   copy values within local atom-based arrays
------------------------------------------------------------------------- */

void FixRigidShell::copy_arrays(int i, int j, int delflag)
{
  bodytag[j] = bodytag[i];
  xcmimage[j] = xcmimage[i];
  displace[j][0] = displace[i][0];
  displace[j][1] = displace[i][1];
  displace[j][2] = displace[i][2];

  if (extended) {
    eflags[j] = eflags[i];
    for (int k = 0; k < orientflag; k++)
      orient[j][k] = orient[i][k];
    if (dorientflag) {
      dorient[j][0] = dorient[i][0];
      dorient[j][1] = dorient[i][1];
      dorient[j][2] = dorient[i][2];
    }
  }

  // must also copy vatom if per-atom virial calculated on this timestep
  // since vatom is calculated before and after atom migration

  if (vflag_atom)
    for (int k = 0; k < 6; k++)
      vatom[j][k] = vatom[i][k];

  // if deleting atom J via delflag and J owns a body, then delete it

  if (delflag && bodyown[j] >= 0) {
    bodyown[body[nlocal_body-1].ilocal] = bodyown[j];
    memcpy(&body[bodyown[j]],&body[nlocal_body-1],sizeof(Body));
    nlocal_body--;
  }

  // if atom I owns a body, reset I's body.ilocal to loc J
  // do NOT do this if self-copy (I=J) since I's body is already deleted

  if (bodyown[i] >= 0 && i != j) body[bodyown[i]].ilocal = j;
  bodyown[j] = bodyown[i];
}

/* ----------------------------------------------------------------------
   initialize one atom's array values, called when atom is created
------------------------------------------------------------------------- */

void FixRigidShell::set_arrays(int i)
{
  bodyown[i] = -1;
  bodytag[i] = 0;
  atom2body[i] = -1;
  angle2body[i] = -1;
  xcmimage[i] = 0;
  displace[i][0] = 0.0;
  displace[i][1] = 0.0;
  displace[i][2] = 0.0;

  vertexdata[i][0] = 0.0;
  vertexdata[i][1] = 0.0;
  vertexdata[i][2] = 0.0;
  vertexdata[i][3] = 0.0;
  vertexdata[i][4] = 0.0;
  vertexdata[i][5] = 0.0;
  vertexdata[i][6] = 0.0;

  // must also zero vatom if per-atom virial calculated on this timestep
  // since vatom is calculated before and after atom migration

  if (vflag_atom)
    for (int k = 0; k < 6; k++)
      vatom[i][k] = 0.0;
}

/* ----------------------------------------------------------------------
   initialize a molecule inserted by another fix, e.g. deposit or pour
   called when molecule is created
   nlocalprev = # of atoms on this proc before molecule inserted
   tagprev = atom ID previous to new atoms in the molecule
   xgeom = geometric center of new molecule
   vcm = COM velocity of new molecule
   quat = rotation of new molecule (around geometric center)
          relative to template in Molecule class
------------------------------------------------------------------------- */

void FixRigidShell::set_molecule(int nlocalprev, tagint tagprev, int imol,
                                 double *xgeom, double *vcm, double *quat)
{
  int m;
  double ctr2com[3],ctr2com_rotate[3];
  double rotmat[3][3];

  // increment total # of rigid bodies

  nbody++;

  // loop over atoms I added for the new body

  int nlocal = atom->nlocal;
  if (nlocalprev == nlocal) return;

  tagint *tag = atom->tag;

  for (int i = nlocalprev; i < nlocal; i++) {
    bodytag[i] = tagprev + onemols[imol]->comatom;
    if (tag[i]-tagprev == onemols[imol]->comatom) bodyown[i] = nlocal_body;

    m = tag[i] - tagprev-1;
    displace[i][0] = onemols[imol]->dxbody[m][0];
    displace[i][1] = onemols[imol]->dxbody[m][1];
    displace[i][2] = onemols[imol]->dxbody[m][2];

    if (extended) {
      eflags[i] = 0;
      if (onemols[imol]->radiusflag) {
        eflags[i] |= SPHERE;
        eflags[i] |= OMEGA;
        eflags[i] |= TORQUE;
      }
    }

    if (bodyown[i] >= 0) {
      if (nlocal_body == nmax_body) grow_body();
      Body *b = &body[nlocal_body];
      b->mass = onemols[imol]->masstotal;
      b->natoms = onemols[imol]->natoms;
      b->xgc[0] = xgeom[0];
      b->xgc[1] = xgeom[1];
      b->xgc[2] = xgeom[2];

      // new COM = Q (onemols[imol]->xcm - onemols[imol]->center) + xgeom
      // Q = rotation matrix associated with quat

      MathExtra::quat_to_mat(quat,rotmat);
      MathExtra::sub3(onemols[imol]->com,onemols[imol]->center,ctr2com);
      MathExtra::matvec(rotmat,ctr2com,ctr2com_rotate);
      MathExtra::add3(ctr2com_rotate,xgeom,b->xcm);

      b->vcm[0] = vcm[0];
      b->vcm[1] = vcm[1];
      b->vcm[2] = vcm[2];
      b->inertia[0] = onemols[imol]->inertia[0];
      b->inertia[1] = onemols[imol]->inertia[1];
      b->inertia[2] = onemols[imol]->inertia[2];

      // final quat is product of insertion quat and original quat
      // true even if insertion rotation was not around COM

      MathExtra::quatquat(quat,onemols[imol]->quat,b->quat);
      MathExtra::q_to_exyz(b->quat,b->ex_space,b->ey_space,b->ez_space);

      MathExtra::transpose_matvec(b->ex_space,b->ey_space,b->ez_space,
                                  ctr2com_rotate,b->xgc_body);
      b->xgc_body[0] *= -1;
      b->xgc_body[1] *= -1;
      b->xgc_body[2] *= -1;

      b->angmom[0] = b->angmom[1] = b->angmom[2] = 0.0;
      b->omega[0] = b->omega[1] = b->omega[2] = 0.0;
      b->conjqm[0] = b->conjqm[1] = b->conjqm[2] = b->conjqm[3] = 0.0;

      b->image = ((imageint) IMGMAX << IMG2BITS) |
        ((imageint) IMGMAX << IMGBITS) | IMGMAX;
      b->ilocal = i;
      nlocal_body++;
    }
  }
}

/* ----------------------------------------------------------------------
   pack values in local atom-based arrays for exchange with another proc
------------------------------------------------------------------------- */

int FixRigidShell::pack_exchange(int i, double *buf)
{
  buf[0] = ubuf(bodytag[i]).d;
  buf[1] = ubuf(xcmimage[i]).d;
  buf[2] = displace[i][0];
  buf[3] = displace[i][1];
  buf[4] = displace[i][2];

  // extended attribute info

  int m = 5;
  if (extended) {
    buf[m++] = eflags[i];
    for (int j = 0; j < orientflag; j++)
      buf[m++] = orient[i][j];
    if (dorientflag) {
      buf[m++] = dorient[i][0];
      buf[m++] = dorient[i][1];
      buf[m++] = dorient[i][2];
    }
  }

  // atom not in a rigid body

  if (!bodytag[i]) return m;

  // must also pack vatom if per-atom virial calculated on this timestep
  // since vatom is calculated before and after atom migration

  if (vflag_atom)
    for (int k = 0; k < 6; k++)
      buf[m++] = vatom[i][k];

  // atom does not own its rigid body

  if (bodyown[i] < 0) {
    buf[m++] = 0;
    return m;
  }

  // body info for atom that owns a rigid body

  buf[m++] = 1;
  memcpy(&buf[m],&body[bodyown[i]],sizeof(Body));
  m += bodysize;
  return m;
}

/* ----------------------------------------------------------------------
   unpack values in local atom-based arrays from exchange with another proc
------------------------------------------------------------------------- */

int FixRigidShell::unpack_exchange(int nlocal, double *buf)
{
  bodytag[nlocal] = (tagint) ubuf(buf[0]).i;
  xcmimage[nlocal] = (imageint) ubuf(buf[1]).i;
  displace[nlocal][0] = buf[2];
  displace[nlocal][1] = buf[3];
  displace[nlocal][2] = buf[4];

  // extended attribute info

  int m = 5;
  if (extended) {
    eflags[nlocal] = static_cast<int> (buf[m++]);
    for (int j = 0; j < orientflag; j++)
      orient[nlocal][j] = buf[m++];
    if (dorientflag) {
      dorient[nlocal][0] = buf[m++];
      dorient[nlocal][1] = buf[m++];
      dorient[nlocal][2] = buf[m++];
    }
  }

  // atom not in a rigid body

  if (!bodytag[nlocal]) {
    bodyown[nlocal] = -1;
    return m;
  }

  // must also unpack vatom if per-atom virial calculated on this timestep
  // since vatom is calculated before and after atom migration

  if (vflag_atom)
    for (int k = 0; k < 6; k++)
      vatom[nlocal][k] = buf[m++];

  // atom does not own its rigid body

  bodyown[nlocal] = static_cast<int> (buf[m++]);
  if (bodyown[nlocal] == 0) {
    bodyown[nlocal] = -1;
    return m;
  }

  // body info for atom that owns a rigid body

  if (nlocal_body == nmax_body) grow_body();
  memcpy(&body[nlocal_body],&buf[m],sizeof(Body));
  m += bodysize;
  body[nlocal_body].ilocal = nlocal;
  bodyown[nlocal] = nlocal_body++;

  return m;
}

/* ----------------------------------------------------------------------
   only pack body info if own or ghost atom owns the body
   for FULL_BODY, send 0/1 flag with every atom
------------------------------------------------------------------------- */

int FixRigidShell::pack_forward_comm(int n, int *list, double *buf,
                                     int /*pbc_flag*/, int * /*pbc*/)
{
  int i,j;
  double *xcm,*xgc,*vcm,*quat,*omega,*ex_space,*ey_space,*ez_space,*conjqm;

  int m = 0;

  if (commflag == INITIAL) {
    for (i = 0; i < n; i++) {
      j = list[i];
      if (bodyown[j] < 0) continue;
      xcm = body[bodyown[j]].xcm;
      buf[m++] = xcm[0];
      buf[m++] = xcm[1];
      buf[m++] = xcm[2];
      xgc = body[bodyown[j]].xgc;
      buf[m++] = xgc[0];
      buf[m++] = xgc[1];
      buf[m++] = xgc[2];
      vcm = body[bodyown[j]].vcm;
      buf[m++] = vcm[0];
      buf[m++] = vcm[1];
      buf[m++] = vcm[2];
      quat = body[bodyown[j]].quat;
      buf[m++] = quat[0];
      buf[m++] = quat[1];
      buf[m++] = quat[2];
      buf[m++] = quat[3];
      omega = body[bodyown[j]].omega;
      buf[m++] = omega[0];
      buf[m++] = omega[1];
      buf[m++] = omega[2];
      ex_space = body[bodyown[j]].ex_space;
      buf[m++] = ex_space[0];
      buf[m++] = ex_space[1];
      buf[m++] = ex_space[2];
      ey_space = body[bodyown[j]].ey_space;
      buf[m++] = ey_space[0];
      buf[m++] = ey_space[1];
      buf[m++] = ey_space[2];
      ez_space = body[bodyown[j]].ez_space;
      buf[m++] = ez_space[0];
      buf[m++] = ez_space[1];
      buf[m++] = ez_space[2];
      conjqm = body[bodyown[j]].conjqm;
      buf[m++] = conjqm[0];
      buf[m++] = conjqm[1];
      buf[m++] = conjqm[2];
      buf[m++] = conjqm[3];
    }

  } else if (commflag == FINAL) {
    for (i = 0; i < n; i++) {
      j = list[i];
      if (bodyown[j] < 0) continue;
      vcm = body[bodyown[j]].vcm;
      buf[m++] = vcm[0];
      buf[m++] = vcm[1];
      buf[m++] = vcm[2];
      omega = body[bodyown[j]].omega;
      buf[m++] = omega[0];
      buf[m++] = omega[1];
      buf[m++] = omega[2];
      conjqm = body[bodyown[j]].conjqm;
      buf[m++] = conjqm[0];
      buf[m++] = conjqm[1];
      buf[m++] = conjqm[2];
      buf[m++] = conjqm[3];
    }

  } else if (commflag == FULL_BODY) {
    for (i = 0; i < n; i++) {
      j = list[i];
      if (bodyown[j] < 0) buf[m++] = 0;
      else {
        buf[m++] = 1;
        memcpy(&buf[m],&body[bodyown[j]],sizeof(Body));
        m += bodysize;
      }
    }
  }

  return m;
}

/* ----------------------------------------------------------------------
   only ghost atoms are looped over
   for FULL_BODY, store a new ghost body if this atom owns it
   for other commflag values, only unpack body info if atom owns it
------------------------------------------------------------------------- */

void FixRigidShell::unpack_forward_comm(int n, int first, double *buf)
{
  int i,j,last;
  double *xcm,*xgc,*vcm,*quat,*omega,*ex_space,*ey_space,*ez_space,*conjqm;

  int m = 0;
  last = first + n;

  if (commflag == INITIAL) {
    for (i = first; i < last; i++) {
      if (bodyown[i] < 0) continue;
      xcm = body[bodyown[i]].xcm;
      xcm[0] = buf[m++];
      xcm[1] = buf[m++];
      xcm[2] = buf[m++];
      xgc = body[bodyown[i]].xgc;
      xgc[0] = buf[m++];
      xgc[1] = buf[m++];
      xgc[2] = buf[m++];
      vcm = body[bodyown[i]].vcm;
      vcm[0] = buf[m++];
      vcm[1] = buf[m++];
      vcm[2] = buf[m++];
      quat = body[bodyown[i]].quat;
      quat[0] = buf[m++];
      quat[1] = buf[m++];
      quat[2] = buf[m++];
      quat[3] = buf[m++];
      omega = body[bodyown[i]].omega;
      omega[0] = buf[m++];
      omega[1] = buf[m++];
      omega[2] = buf[m++];
      ex_space = body[bodyown[i]].ex_space;
      ex_space[0] = buf[m++];
      ex_space[1] = buf[m++];
      ex_space[2] = buf[m++];
      ey_space = body[bodyown[i]].ey_space;
      ey_space[0] = buf[m++];
      ey_space[1] = buf[m++];
      ey_space[2] = buf[m++];
      ez_space = body[bodyown[i]].ez_space;
      ez_space[0] = buf[m++];
      ez_space[1] = buf[m++];
      ez_space[2] = buf[m++];
      conjqm = body[bodyown[i]].conjqm;
      conjqm[0] = buf[m++];
      conjqm[1] = buf[m++];
      conjqm[2] = buf[m++];
      conjqm[3] = buf[m++];
    }

  } else if (commflag == FINAL) {
    for (i = first; i < last; i++) {
      if (bodyown[i] < 0) continue;
      vcm = body[bodyown[i]].vcm;
      vcm[0] = buf[m++];
      vcm[1] = buf[m++];
      vcm[2] = buf[m++];
      omega = body[bodyown[i]].omega;
      omega[0] = buf[m++];
      omega[1] = buf[m++];
      omega[2] = buf[m++];
      conjqm = body[bodyown[i]].conjqm;
      conjqm[0] = buf[m++];
      conjqm[1] = buf[m++];
      conjqm[2] = buf[m++];
      conjqm[3] = buf[m++];
    }

  } else if (commflag == FULL_BODY) {
    for (i = first; i < last; i++) {
      bodyown[i] = static_cast<int> (buf[m++]);
      if (bodyown[i] == 0) bodyown[i] = -1;
      else {
        j = nlocal_body + nghost_body;
        if (j == nmax_body) grow_body();
        memcpy(&body[j],&buf[m],sizeof(Body));
        m += bodysize;
        body[j].ilocal = i;
        bodyown[i] = j;
        nghost_body++;
      }
    }
  }
}

/* ----------------------------------------------------------------------
   only ghost atoms are looped over
   only pack body info if atom owns it
------------------------------------------------------------------------- */

int FixRigidShell::pack_reverse_comm(int n, int first, double *buf)
{
  int i,j,m,last;
  double *fcm,*torque,*vcm,*angmom,*xcm, *xgc;

  m = 0;
  last = first + n;

  if (commflag == FORCE_TORQUE) {
    for (i = first; i < last; i++) {
      if (bodyown[i] < 0) continue;
      fcm = body[bodyown[i]].fcm;
      buf[m++] = fcm[0];
      buf[m++] = fcm[1];
      buf[m++] = fcm[2];
      torque = body[bodyown[i]].torque;
      buf[m++] = torque[0];
      buf[m++] = torque[1];
      buf[m++] = torque[2];
    }

  } else if (commflag == VCM_ANGMOM) {
    for (i = first; i < last; i++) {
      if (bodyown[i] < 0) continue;
      vcm = body[bodyown[i]].vcm;
      buf[m++] = vcm[0];
      buf[m++] = vcm[1];
      buf[m++] = vcm[2];
      angmom = body[bodyown[i]].angmom;
      buf[m++] = angmom[0];
      buf[m++] = angmom[1];
      buf[m++] = angmom[2];
    }

  } else if (commflag == XCM_MASS) {
    for (i = first; i < last; i++) {
      if (bodyown[i] < 0) continue;
      xcm = body[bodyown[i]].xcm;
      xgc = body[bodyown[i]].xgc;
      buf[m++] = xcm[0];
      buf[m++] = xcm[1];
      buf[m++] = xcm[2];
      buf[m++] = xgc[0];
      buf[m++] = xgc[1];
      buf[m++] = xgc[2];
      buf[m++] = body[bodyown[i]].mass;
      buf[m++] = static_cast<double>(body[bodyown[i]].natoms);
    }

  } else if (commflag == ITENSOR) {
    for (i = first; i < last; i++) {
      if (bodyown[i] < 0) continue;
      j = bodyown[i];
      buf[m++] = itensor[j][0];
      buf[m++] = itensor[j][1];
      buf[m++] = itensor[j][2];
      buf[m++] = itensor[j][3];
      buf[m++] = itensor[j][4];
      buf[m++] = itensor[j][5];
    }

  } else if (commflag == DOF) {
    for (i = first; i < last; i++) {
      if (bodyown[i] < 0) continue;
      j = bodyown[i];
      buf[m++] = counts[j][0];
      buf[m++] = counts[j][1];
      buf[m++] = counts[j][2];
    }
  }

  return m;
}

/* ----------------------------------------------------------------------
   only unpack body info if own or ghost atom owns the body
------------------------------------------------------------------------- */

void FixRigidShell::unpack_reverse_comm(int n, int *list, double *buf)
{
  int i,j,k;
  double *fcm,*torque,*vcm,*angmom,*xcm, *xgc;

  int m = 0;

  if (commflag == FORCE_TORQUE) {
    for (i = 0; i < n; i++) {
      j = list[i];
      if (bodyown[j] < 0) continue;
      fcm = body[bodyown[j]].fcm;
      fcm[0] += buf[m++];
      fcm[1] += buf[m++];
      fcm[2] += buf[m++];
      torque = body[bodyown[j]].torque;
      torque[0] += buf[m++];
      torque[1] += buf[m++];
      torque[2] += buf[m++];
    }

  } else if (commflag == VCM_ANGMOM) {
    for (i = 0; i < n; i++) {
      j = list[i];
      if (bodyown[j] < 0) continue;
      vcm = body[bodyown[j]].vcm;
      vcm[0] += buf[m++];
      vcm[1] += buf[m++];
      vcm[2] += buf[m++];
      angmom = body[bodyown[j]].angmom;
      angmom[0] += buf[m++];
      angmom[1] += buf[m++];
      angmom[2] += buf[m++];
    }

  } else if (commflag == XCM_MASS) {
    for (i = 0; i < n; i++) {
      j = list[i];
      if (bodyown[j] < 0) continue;
      xcm = body[bodyown[j]].xcm;
      xgc = body[bodyown[j]].xgc;
      xcm[0] += buf[m++];
      xcm[1] += buf[m++];
      xcm[2] += buf[m++];
      xgc[0] += buf[m++];
      xgc[1] += buf[m++];
      xgc[2] += buf[m++];
      body[bodyown[j]].mass += buf[m++];
      body[bodyown[j]].natoms += static_cast<int>(buf[m++]);
    }

  } else if (commflag == ITENSOR) {
    for (i = 0; i < n; i++) {
      j = list[i];
      if (bodyown[j] < 0) continue;
      k = bodyown[j];
      itensor[k][0] += buf[m++];
      itensor[k][1] += buf[m++];
      itensor[k][2] += buf[m++];
      itensor[k][3] += buf[m++];
      itensor[k][4] += buf[m++];
      itensor[k][5] += buf[m++];
    }

  } else if (commflag == DOF) {
    for (i = 0; i < n; i++) {
      j = list[i];
      if (bodyown[j] < 0) continue;
      k = bodyown[j];
      counts[k][0] += static_cast<int> (buf[m++]);
      counts[k][1] += static_cast<int> (buf[m++]);
      counts[k][2] += static_cast<int> (buf[m++]);
    }
  }
}

/* ----------------------------------------------------------------------
   grow body data structure
------------------------------------------------------------------------- */

void FixRigidShell::grow_body()
{
  nmax_body += DELTA_BODY;
  body = (Body *) memory->srealloc(body,nmax_body*sizeof(Body),
                                   "rigid/shell:body");
}

/* ----------------------------------------------------------------------
   reset atom2body for all owned atoms
   do this via bodyown of atom that owns the body the owned atom is in
   atom2body values can point to original body or any image of the body
------------------------------------------------------------------------- */

void FixRigidShell::reset_atom2body()
{
  int iowner;

  // iowner = index of atom that owns the body that atom I is in

  int nlocal = atom->nlocal;

  for (int i = 0; i < nlocal; i++) {
    atom2body[i] = -1;
    if (bodytag[i]) {
      iowner = atom->map(bodytag[i]);
      if (iowner == -1)
        error->one(FLERR,"Rigid body atoms {} {} missing on "
                                     "proc {} at step {}",atom->tag[i],
                                     bodytag[i],comm->me,update->ntimestep);

      atom2body[i] = bodyown[iowner];
    }
  }
}

/* ---------------------------------------------------------------------- */

/* ----------------------------------------------------------------------
   reset angle2body for all owned angles
   do this via bodyown of atom, in the angle, that owns the body the owned atom is in
   angle2body values can point to original body or any image of the body
------------------------------------------------------------------------- */

void FixRigidShell::reset_angle2body()
{
  int iowner;

  // iowner = index of atom that owns the body that atom I is in

  int nanglelist = neighbor->nanglelist;
  int **anglelist = neighbor->anglelist;

  for (int n = 0; n < nanglelist; n++) {
    angle2body[n] = -1;

    if (bodytag[anglelist[n][0]]) {
      iowner = atom->map(bodytag[anglelist[n][0]]);
      if (iowner == -1)
        error->one(FLERR,"Rigid body atoms {} {} missing on "
                                     "proc {} at step {}",atom->tag[anglelist[n][0]],
                                     bodytag[anglelist[n][0]],comm->me,update->ntimestep);

      angle2body[n] = bodyown[iowner];
    }
  }
}

/* ---------------------------------------------------------------------- */
void FixRigidShell::reset_dt()
{
  dtv = update->dt;
  dtf = 0.5 * update->dt * force->ftm2v;
  dtq = 0.5 * update->dt;
}

/* ----------------------------------------------------------------------
   zero linear momentum of each rigid body
   set Vcm to 0.0, then reset velocities of particles via set_v()
------------------------------------------------------------------------- */

void FixRigidShell::zero_momentum()
{
  double *vcm;
  for (int ibody = 0; ibody < nlocal_body+nghost_body; ibody++) {
    vcm = body[ibody].vcm;
    vcm[0] = vcm[1] = vcm[2] = 0.0;
  }

  // forward communicate of vcm to all ghost copies

  commflag = FINAL;
  comm->forward_comm(this,10);

  // set velocity of atoms in rigid bodues

  evflag = 0;
  set_v();
}

/* ----------------------------------------------------------------------
   zero angular momentum of each rigid body
   set angmom/omega to 0.0, then reset velocities of particles via set_v()
------------------------------------------------------------------------- */

void FixRigidShell::zero_rotation()
{
  double *angmom,*omega;
  for (int ibody = 0; ibody < nlocal_body+nghost_body; ibody++) {
    angmom = body[ibody].angmom;
    angmom[0] = angmom[1] = angmom[2] = 0.0;
    omega = body[ibody].omega;
    omega[0] = omega[1] = omega[2] = 0.0;
  }

  // forward communicate of omega to all ghost copies

  commflag = FINAL;
  comm->forward_comm(this,10);

  // set velocity of atoms in rigid bodues

  evflag = 0;
  set_v();
}

/* ---------------------------------------------------------------------- */

int FixRigidShell::modify_param(int narg, char **arg)
{
  if (strcmp(arg[0],"bodyforces") == 0) {
    if (narg < 2) error->all(FLERR,"Illegal fix_modify command");
    if (strcmp(arg[1],"early") == 0) earlyflag = 1;
    else if (strcmp(arg[1],"late") == 0) earlyflag = 0;
    else error->all(FLERR,"Illegal fix_modify command");

    // reset fix mask
    // must do here and not in init,
    // since modify.cpp::init() uses fix masks before calling fix::init()

    for (int i = 0; i < modify->nfix; i++)
      if (strcmp(modify->fix[i]->id,id) == 0) {
        if (earlyflag) modify->fmask[i] |= POST_FORCE;
        else if (!langflag) modify->fmask[i] &= ~POST_FORCE;
        break;
      }

    return 2;
  }

  return 0;
}

/* ---------------------------------------------------------------------- */

void *FixRigidShell::extract(const char *str, int &dim)
{
  dim = 0;

  if (strcmp(str,"body") == 0) {
    if (!setupflag) return nullptr;
    dim = 1;
    return atom2body;
  }

  if (strcmp(str,"onemol") == 0) {
    dim = 0;
    return onemols;
  }

  // return vector of rigid body masses, for owned+ghost bodies
  // used by granular pair styles, indexed by atom2body

  if (strcmp(str,"masstotal") == 0) {
    if (!setupflag) return nullptr;
    dim = 1;

    if (nmax_mass < nmax_body) {
      memory->destroy(mass_body);
      nmax_mass = nmax_body;
      memory->create(mass_body,nmax_mass,"rigid:mass_body");
    }

    int n = nlocal_body + nghost_body;
    for (int i = 0; i < n; i++)
      mass_body[i] = body[i].mass;

    return mass_body;
  }

  return nullptr;
}

/* ----------------------------------------------------------------------
   return translational KE for all rigid bodies
   KE = 1/2 M Vcm^2
   sum local body results across procs
------------------------------------------------------------------------- */

double FixRigidShell::extract_ke()
{
  double *vcm;

  double ke = 0.0;
  for (int i = 0; i < nlocal_body; i++) {
    vcm = body[i].vcm;
    ke += body[i].mass * (vcm[0]*vcm[0] + vcm[1]*vcm[1] + vcm[2]*vcm[2]);
  }

  double keall;
  MPI_Allreduce(&ke,&keall,1,MPI_DOUBLE,MPI_SUM,world);

  return 0.5*keall;
}

/* ----------------------------------------------------------------------
   return rotational KE for all rigid bodies
   Erotational = 1/2 I wbody^2
------------------------------------------------------------------------- */

double FixRigidShell::extract_erotational()
{
  double wbody[3],rot[3][3];
  double *inertia;

  double erotate = 0.0;
  for (int i = 0; i < nlocal_body; i++) {

    // for Iw^2 rotational term, need wbody = angular velocity in body frame
    // not omega = angular velocity in space frame

    inertia = body[i].inertia;
    MathExtra::quat_to_mat(body[i].quat,rot);
    MathExtra::transpose_matvec(rot,body[i].angmom,wbody);
    if (inertia[0] == 0.0) wbody[0] = 0.0;
    else wbody[0] /= inertia[0];
    if (inertia[1] == 0.0) wbody[1] = 0.0;
    else wbody[1] /= inertia[1];
    if (inertia[2] == 0.0) wbody[2] = 0.0;
    else wbody[2] /= inertia[2];

    erotate += inertia[0]*wbody[0]*wbody[0] + inertia[1]*wbody[1]*wbody[1] +
      inertia[2]*wbody[2]*wbody[2];
  }

  double erotateall;
  MPI_Allreduce(&erotate,&erotateall,1,MPI_DOUBLE,MPI_SUM,world);

  return 0.5*erotateall;
}

/* ----------------------------------------------------------------------
   return temperature of collection of rigid bodies
   non-active DOF are removed by fflag/tflag and in tfactor
------------------------------------------------------------------------- */

double FixRigidShell::compute_scalar()
{
  double wbody[3],rot[3][3];

  double *vcm,*inertia;

  double t = 0.0;

  for (int i = 0; i < nlocal_body; i++) {
    vcm = body[i].vcm;
    t += body[i].mass * (vcm[0]*vcm[0] + vcm[1]*vcm[1] + vcm[2]*vcm[2]);

    // for Iw^2 rotational term, need wbody = angular velocity in body frame
    // not omega = angular velocity in space frame

    inertia = body[i].inertia;
    MathExtra::quat_to_mat(body[i].quat,rot);
    MathExtra::transpose_matvec(rot,body[i].angmom,wbody);
    if (inertia[0] == 0.0) wbody[0] = 0.0;
    else wbody[0] /= inertia[0];
    if (inertia[1] == 0.0) wbody[1] = 0.0;
    else wbody[1] /= inertia[1];
    if (inertia[2] == 0.0) wbody[2] = 0.0;
    else wbody[2] /= inertia[2];

    t += inertia[0]*wbody[0]*wbody[0] + inertia[1]*wbody[1]*wbody[1] +
      inertia[2]*wbody[2]*wbody[2];
  }

  double tall;
  MPI_Allreduce(&t,&tall,1,MPI_DOUBLE,MPI_SUM,world);

  double tfactor = force->mvv2e / ((6.0*nbody - nlinear) * force->boltz);
  tall *= tfactor;
  return tall;
}

/* ----------------------------------------------------------------------
   memory usage of local atom-based arrays
------------------------------------------------------------------------- */

double FixRigidShell::memory_usage()
{
  int nmax = atom->nmax;
  double bytes = (double)nmax*2 * sizeof(int);
  bytes += (double)nmax * sizeof(imageint);
  bytes += (double)nmax*3 * sizeof(double);
  bytes += (double)maxvatom*6 * sizeof(double);     // vatom
  if (extended) {
    bytes += (double)nmax * sizeof(int);
    if (orientflag) bytes = (double)nmax*orientflag * sizeof(double);
    if (dorientflag) bytes = (double)nmax*3 * sizeof(double);
  }
  bytes += (double)nmax_body * sizeof(Body);
  return bytes;
}

/* ----------------------------------------------------------------------
   debug method for sanity checking of atom/body data pointers
------------------------------------------------------------------------- */

/*
void FixRigidShell::check(int flag)
{
  for (int i = 0; i < atom->nlocal; i++) {
    if (bodyown[i] >= 0) {
      if (bodytag[i] != atom->tag[i]) {
        printf("Proc %d, step %ld, flag %d\n",comm->me,update->ntimestep,flag);
        errorx->one(FLERR,"BAD AAA");
      }
      if (bodyown[i] < 0 || bodyown[i] >= nlocal_body) {
        printf("Proc %d, step %ld, flag %d\n",comm->me,update->ntimestep,flag);
        errorx->one(FLERR,"BAD BBB");
      }
      if (atom2body[i] != bodyown[i]) {
        printf("Proc %d, step %ld, flag %d\n",comm->me,update->ntimestep,flag);
        errorx->one(FLERR,"BAD CCC");
      }
      if (body[bodyown[i]].ilocal != i) {
        printf("Proc %d, step %ld, flag %d\n",comm->me,update->ntimestep,flag);
        errorx->one(FLERR,"BAD DDD");
      }
    }
  }

  for (int i = 0; i < atom->nlocal; i++) {
    if (bodyown[i] < 0 && bodytag[i] > 0) {
      if (atom2body[i] < 0 || atom2body[i] >= nlocal_body+nghost_body) {
        printf("Proc %d, step %ld, flag %d\n",comm->me,update->ntimestep,flag);
        errorx->one(FLERR,"BAD EEE");
      }
      if (bodytag[i] != atom->tag[body[atom2body[i]].ilocal]) {
        printf("Proc %d, step %ld, flag %d\n",comm->me,update->ntimestep,flag);
        errorx->one(FLERR,"BAD FFF");
      }
    }
  }

  for (int i = atom->nlocal; i < atom->nlocal + atom->nghost; i++) {
    if (bodyown[i] >= 0) {
      if (bodyown[i] < nlocal_body ||
          bodyown[i] >= nlocal_body+nghost_body) {
        printf("Values %d %d: %d %d %d\n",
               i,atom->tag[i],bodyown[i],nlocal_body,nghost_body);
        printf("Proc %d, step %ld, flag %d\n",comm->me,update->ntimestep,flag);
        errorx->one(FLERR,"BAD GGG");
      }
      if (body[bodyown[i]].ilocal != i) {
        printf("Proc %d, step %ld, flag %d\n",comm->me,update->ntimestep,flag);
        errorx->one(FLERR,"BAD HHH");
      }
    }
  }

  for (int i = 0; i < nlocal_body; i++) {
    if (body[i].ilocal < 0 || body[i].ilocal >= atom->nlocal) {
      printf("Proc %d, step %ld, flag %d\n",comm->me,update->ntimestep,flag);
      errorx->one(FLERR,"BAD III");
    }
    if (bodytag[body[i].ilocal] != atom->tag[body[i].ilocal] ||
        bodyown[body[i].ilocal] != i) {
      printf("Proc %d, step %ld, flag %d\n",comm->me,update->ntimestep,flag);
      errorx->one(FLERR,"BAD JJJ");
    }
  }

  for (int i = nlocal_body; i < nlocal_body + nghost_body; i++) {
    if (body[i].ilocal < atom->nlocal ||
        body[i].ilocal >= atom->nlocal + atom->nghost) {
      printf("Proc %d, step %ld, flag %d\n",comm->me,update->ntimestep,flag);
      errorx->one(FLERR,"BAD KKK");
    }
    if (bodyown[body[i].ilocal] != i) {
      printf("Proc %d, step %ld, flag %d\n",comm->me,update->ntimestep,flag);
      errorx->one(FLERR,"BAD LLL");
    }
  }
}
*/
